import axios from 'axios';
import React, { useCallback, useEffect, useState, useRef } from 'react';
import { FaSpinner } from 'react-icons/fa';
import { Document, DocumentSuggestion } from './DocumentProcessor';
import { Tooltip } from 'react-tooltip';

type OCRPageResult = {
  text: string;
  ocrLimitHit: boolean;
  generationInfo?: Record<string, any>;
};

const ExperimentalOCR: React.FC = () => {
  const refreshInterval = 1000; // Refresh interval in milliseconds
  const [documentId, setDocumentId] = useState(0);
  const [jobId, setJobId] = useState('');
  const [ocrResult, setOcrResult] = useState('');
  const [status, setStatus] = useState('');
  const [error, setError] = useState<string | null>('');
  const [pagesDone, setPagesDone] = useState(0);
  const [saving, setSaving] = useState(false);
  const [documentDetails, setDocumentDetails] = useState<Document | null>(null);
  const [perPageResults, setPerPageResults] = useState<OCRPageResult[]>([]);
  const lastFetchedPagesDoneRef = useRef(0);
  
  const [reOcrLoading, setReOcrLoading] = useState<{ [pageIdx: number]: boolean }>({});
  const [reOcrErrors, setReOcrErrors] = useState<{ [pageIdx: number]: string }>({});

  const fetchDocumentDetails = useCallback(async () => {
    if (!documentId) return;

    try {
      const response = await axios.get<Document>(`/api/documents/${documentId}`);
      setDocumentDetails(response.data);
    } catch (err) {
      console.error("Error fetching document details:", err);
      setError("Failed to fetch document details.");
    }
  }, [documentId]);

  const fetchPerPageResults = useCallback(async () => {
    if (!documentId) return;
    try {
      const response = await axios.get<{ pages: OCRPageResult[] }>(`/api/documents/${documentId}/ocr_pages`);
      setPerPageResults(response.data.pages);
    } catch (err) {
      console.error("Error fetching per-page OCR results:", err);
      setError("Failed to fetch per-page OCR results.");
    }
  }, [documentId]);

  const submitOCRJob = async () => {
    setStatus('');
    setError('');
    setJobId('');
    setOcrResult('');
    setPagesDone(0);
    setPerPageResults([]);
    lastFetchedPagesDoneRef.current = 0;

    try {
      setStatus('Fetching document details...');
      await fetchDocumentDetails();

      setStatus('Submitting OCR job...');
      const response = await axios.post(`/api/documents/${documentId}/ocr`);
      setJobId(response.data.job_id);
      setStatus('Job submitted. Processing...');
    } catch (err) {
      console.error(err);
      setError('Failed to submit OCR job.');
    }
  };

  const checkJobStatus = async () => {
    if (!jobId) return;

    try {
      const response = await axios.get(`/api/jobs/ocr/${jobId}`);
      const jobStatus = response.data.status;
      const newPagesDone = response.data.pages_done;
      setPagesDone(newPagesDone);

      // Only fetch per-page results if new pages are done
      if (newPagesDone > lastFetchedPagesDoneRef.current) {
        await fetchPerPageResults();
        lastFetchedPagesDoneRef.current = newPagesDone;
      }

      if (jobStatus === 'completed') {
        // Parse the result as JSON
        let parsedResult: { combinedText: string; perPageResults: OCRPageResult[] } | null = null;
        try {
          parsedResult = JSON.parse(response.data.result);
        } catch (e) {
          setOcrResult(response.data.result);
          setStatus('OCR completed successfully.');
          return;
        }
        if (parsedResult) {
          setOcrResult(parsedResult.combinedText);
          setPerPageResults(parsedResult.perPageResults);
        }
        setStatus('OCR completed successfully.');
      } else if (jobStatus === 'failed') {
        setError(response.data.error);
        setStatus('OCR failed.');
      } else {
        setStatus(`Job status: ${jobStatus}. This may take a few minutes.`);
        setTimeout(() => checkJobStatus(), refreshInterval);
      }
    } catch (err) {
      console.error(err);
      setError('Failed to check job status.');
    }
  };

  const handleSaveContent = async () => {
    setSaving(true);
    setError(null);
    try {
      if (!documentDetails) {
        setError('Document details not fetched.');
        throw new Error('Document details not fetched.');
      }
      const requestPayload: DocumentSuggestion = {
        id: documentId,
        original_document: documentDetails,
        suggested_content: ocrResult,
      };

      await axios.patch("/api/update-documents", [requestPayload]);
      setStatus('Content saved successfully.');
    } catch (err) {
      console.error("Error saving content:", err);
      setError("Failed to save content.");
    } finally {
      setSaving(false);
    }
  };

  // Re-OCR a single page
  const handleReOcrPage = async (pageIdx: number) => {
    if (!perPageResults[pageIdx]) {
      setReOcrErrors((prev) => ({ ...prev, [pageIdx]: "Page data not available." }));
      return;
    }

    setReOcrLoading((prev) => ({ ...prev, [pageIdx]: true }));
    setReOcrErrors((prev) => ({ ...prev, [pageIdx]: "" }));

    try {
      const response = await axios.post(`/api/documents/${documentId}/ocr_pages/${pageIdx}/reocr`);
      // Update the perPageResults for this page
      setPerPageResults((prev) =>
        prev.map((res, idx) =>
          idx === pageIdx
            ? {
                text: response.data.text,
                ocrLimitHit: response.data.ocrLimitHit,
              }
            : res
        )
      );
      // After re-OCR, also update lastFetchedPagesDone if this page was not previously available
      if (pageIdx + 1 > lastFetchedPagesDoneRef.current) {
        lastFetchedPagesDoneRef.current = pageIdx + 1;
      }
    } catch (err) {
      setReOcrErrors((prev) => ({
        ...prev,
        [pageIdx]: "Failed to re-OCR page.",
      }));
    } finally {
      setReOcrLoading((prev) => ({ ...prev, [pageIdx]: false }));
    }
  };

  useEffect(() => {
    if (jobId) {
      lastFetchedPagesDoneRef.current = 0; // Reset when new job starts
      checkJobStatus();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  return (
    <div className="max-w-3xl mx-auto p-6 bg-white dark:bg-gray-900 text-gray-800 dark:text-gray-200">
      <h1 className="text-4xl font-bold mb-6 text-center">OCR via LLMs (Experimental)</h1>
      <p className="mb-6 text-center text-yellow-600">
        This is an experimental feature. Results may vary, and processing may take some time.
      </p>
      <div className="bg-gray-100 dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <div className="mb-4">
          <label htmlFor="documentId" className="block mb-2 font-semibold">
            Document ID:
          </label>
          <input
            type="number"
            id="documentId"
            value={documentId}
            onChange={(e) => setDocumentId(Number(e.target.value))}
            className="border border-gray-300 dark:border-gray-700 rounded w-full p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter the document ID"
          />
        </div>
        <button
          onClick={submitOCRJob}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded transition duration-200"
          disabled={!documentId}
        >
          {status.startsWith('Submitting') ? (
            <span className="flex items-center justify-center">
              <FaSpinner className="animate-spin mr-2" />
              Submitting...
            </span>
          ) : (
            'Submit OCR Job'
          )}
        </button>
        {status && (
          <div className="mt-4 text-center text-gray-700 dark:text-gray-300">
            {status.includes('in_progress') && (
              <span className="flex items-center justify-center">
                <FaSpinner className="animate-spin mr-2" />
                {status}
              </span>
            )}
            {!status.includes('in_progress') && status}
            {pagesDone > 0 && (
              <div className="mt-2">
                Pages processed: {pagesDone}
              </div>
            )}
          </div>
        )}
        {error && (
          <div className="mt-4 p-4 bg-red-100 dark:bg-red-800 text-red-700 dark:text-red-200 rounded">
            {error}
          </div>
        )}
        {perPageResults.length > 0 && (
          <div className="mt-6">
            <h2 className="text-2xl font-bold mb-4">Per-Page OCR Results:</h2>
            {perPageResults.map((page, idx) => (
              <div key={idx} className="mb-6 border border-gray-300 dark:border-gray-700 rounded p-4 bg-white dark:bg-gray-900">
                <div className="flex items-center mb-2">
                  <span className="font-semibold mr-2">Page {idx + 1}</span>
                  {page.ocrLimitHit && (
                    <span className="ml-2 px-2 py-1 bg-yellow-200 text-yellow-800 rounded text-xs font-bold">
                      Token Limit Hit
                    </span>
                  )}
                  {page.generationInfo && Object.keys(page.generationInfo).length > 0 && (
                    <>
                      <span
                        data-tooltip-id={`geninfo-tooltip-${idx}`}
                        className="ml-3 cursor-pointer text-blue-600 hover:text-blue-800"
                        tabIndex={0}
                        aria-label="Show Generation Info"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="inline-block" width="18" height="18" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-12.25a.75.75 0 11-1.5 0 .75.75 0 011.5 0zM9 9.25A1 1 0 0110 8.5h.01a1 1 0 01.99 1v4a1 1 0 01-2 0v-4z"/>
                        </svg>
                      </span>
                      <Tooltip
                        id={`geninfo-tooltip-${idx}`}
                        place="top"
                        className="!max-w-xs !text-xs"
                        style={{ zIndex: 9999 }}
                        clickable={true}
                        render={() => (
                          <div className="p-1">
                            <table>
                              <tbody>
                                {Object.entries(page.generationInfo ?? {}).map(([key, value]) => (
                                  <tr key={key}>
                                    <td className="pr-2 font-semibold align-top">{key}:</td>
                                    <td className="break-all">{typeof value === 'object' ? JSON.stringify(value) : String(value)}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      />
                    </>
                  )}
                </div>
                <pre className="whitespace-pre-wrap bg-gray-50 dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700 overflow-auto max-h-48">
                  {page.text}
                </pre>
                <div className="mt-2 flex flex-col sm:flex-row items-start sm:items-center gap-2">
                  <button
                    onClick={() => handleReOcrPage(idx)}
                    className="bg-orange-600 hover:bg-orange-700 text-white font-semibold py-2 px-4 rounded transition duration-200"
                    disabled={reOcrLoading[idx]}
                  >
                    {reOcrLoading[idx] ? (
                      <span className="flex items-center">
                        <FaSpinner className="animate-spin mr-2" />
                        Re-OCRing...
                      </span>
                    ) : (
                      'Re-OCR Page'
                    )}
                  </button>
                  {reOcrErrors[idx] && (
                    <span className="text-red-600 text-sm ml-2">{reOcrErrors[idx]}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        {ocrResult && (
          <div className="mt-6">
            <h2 className="text-2xl font-bold mb-4">Combined OCR Result:</h2>
            <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded border border-gray-200 dark:border-gray-700 overflow-auto max-h-96">
              <pre className="whitespace-pre-wrap">{ocrResult}</pre>
            </div>
            <button
              onClick={handleSaveContent}
              className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded transition duration-200 mt-4"
              disabled={saving}
            >
              {saving ? (
                <span className="flex items-center justify-center">
                  <FaSpinner className="animate-spin mr-2" />
                  Saving...
                </span>
              ) : (
                'Save Content'
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ExperimentalOCR;
