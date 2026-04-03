    reader.readAsText(selectedFile);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold leading-7 text-neutral-900 dark:text-white sm:truncate">
          Batch Upload Risk Scoring
        </h2>
        <button className="bg-white border border-neutral-300 dark:border-neutral-700 text-neutral-700 dark:text-neutral-300 px-4 py-2 rounded-md text-sm font-medium hover:bg-neutral-50 dark:hover:bg-neutral-800">
          Download CSV Template
        </button>
      </div>

      <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 p-8 text-center">
        {status === "idle" && (
          <div
            className={`mt-2 flex justify-center rounded-lg border-2 border-dashed px-6 py-16 transition-colors ${
              isDragging ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20" : "border-neutral-300 dark:border-neutral-700 hover:border-blue-400"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="text-center">
              <UploadCloud className="mx-auto h-12 w-12 text-neutral-300 dark:text-neutral-600" aria-hidden="true" />
              <div className="mt-4 flex text-sm leading-6 text-neutral-600 dark:text-neutral-400 justify-center">
                <label
                  htmlFor="file-upload"
                  className="relative cursor-pointer rounded-md bg-white dark:bg-neutral-900 font-semibold text-blue-600 focus-within:outline-none focus-within:ring-2 focus-within:ring-blue-600 focus-within:ring-offset-2 hover:text-blue-500"
                >
                  <span>Upload a file</span>
                  <input
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    accept=".csv"
                    className="sr-only"
                    ref={fileInputRef}
                    onChange={(e) => {
                      if (e.target.files && e.target.files.length > 0) {
                        processFile(e.target.files[0]);
                      }
                    }}
                  />
                </label>
                <p className="pl-1">or drag and drop</p>
              </div>
              <p className="text-xs leading-5 text-neutral-500 dark:text-neutral-500 mt-2">CSV up to 10MB (max 10,000 rows)</p>
            </div>
          </div>
        )}

        {(status === "uploading" || status === "processing") && (
          <div className="py-16 flex flex-col items-center">
            <Loader2 className="h-10 w-10 text-blue-500 animate-spin mb-4" />
            <p className="text-lg font-medium text-neutral-900 dark:text-white">
              {status === "uploading" ? "Uploading dataset..." : "Running Risk Models..."}
            </p>
            <div className="w-64 bg-neutral-200 dark:bg-neutral-800 rounded-full h-2.5 mt-4 overflow-hidden">
              <div 
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300 ease-out" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        )}

        {status === "success" && (
          <div className="py-10">
             <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-900/30 mb-4">
              <CheckCircle2 className="h-10 w-10 text-emerald-600 dark:text-emerald-400" />
            </div>
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-white">Processing Complete</h3>
            <p className="text-neutral-500 dark:text-neutral-400 mt-2 mb-6">Successfully analyzed {results.length} borrower records from {file?.name}</p>
            <div className="flex justify-center space-x-4">
              <button 
                onClick={() => setStatus("idle")}
                className="bg-white border border-neutral-300 dark:border-neutral-700 text-neutral-700 dark:text-neutral-300 px-4 py-2 rounded-md text-sm font-medium hover:bg-neutral-50 dark:hover:bg-neutral-800"
              >
                Upload Another File
              </button>
              <button className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
                Export Results (CSV)
              </button>
            </div>
          </div>
        )}
      </div>

      {status === "success" && (
         <div className="bg-white dark:bg-neutral-900 shadow-sm rounded-xl border border-neutral-200 dark:border-neutral-800 overflow-hidden">
            <div className="px-6 py-4 border-b border-neutral-200 dark:border-neutral-800 flex justify-between items-center">
              <h3 className="text-lg font-medium text-neutral-900 dark:text-white">Batch Results</h3>
              <span className="bg-blue-100 text-blue-800 text-xs font-semibold px-2.5 py-0.5 rounded dark:bg-blue-900/30 dark:text-blue-300">
                {results.length} Auto-Assigned
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-neutral-200 dark:divide-neutral-800">
                <thead className="bg-neutral-50 dark:bg-neutral-800/50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Borrower</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Risk Score</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Exp. Loss</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Recommended Action</th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">Assigned Team</th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-neutral-900 divide-y divide-neutral-200 dark:divide-neutral-800">
                  {results.map((row) => (
                    <tr key={row.id} className="hover:bg-neutral-50 dark:hover:bg-neutral-800/50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-neutral-900 dark:text-white">{row.borrower}</td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <span className={`text-sm font-semibold mr-2 ${
                            row.score > 75 ? 'text-rose-600' : row.score > 50 ? 'text-orange-500' : 'text-emerald-500'
                          }`}>{row.score}</span>
                          <div className="w-16 bg-neutral-200 dark:bg-neutral-700 rounded-full h-1.5">
                            <div className={`h-1.5 rounded-full ${
                              row.score > 75 ? 'bg-rose-500' : row.score > 50 ? 'bg-orange-400' : 'bg-emerald-500'
                            }`} style={{ width: `${row.score}%` }}></div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-500 dark:text-neutral-400">{row.expLoss}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-neutral-900 dark:text-neutral-300 flex items-center">
                        {row.score > 80 && <AlertTriangle className="mr-1.5 h-4 w-4 text-rose-500" />}
                        {row.action}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                         <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-neutral-100 text-neutral-800 dark:bg-neutral-800 dark:text-neutral-300">
                          {row.team}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
         </div>
      )}
    </div>
  );
}
