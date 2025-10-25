import React, { useState, useRef, DragEvent, useCallback, memo } from "react";
import { Upload, ArrowUp } from "lucide-react";

interface FileUploaderProps {
  onSubmit: (data: { url?: string; file?: File }) => void;
}

const FileUploader: React.FC<FileUploaderProps> = memo(({ onSubmit }) => {
  const [url, setUrl] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragEnter = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type === "application/pdf") {
        setSelectedFile(file);
        setUrl("");
      } else {
        alert("Please upload a PDF file");
      }
    }
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (file.type === "application/pdf") {
        setSelectedFile(file);
        setUrl("");
      } else {
        alert("Please upload a PDF file");
      }
    }
  }, []);

  const handleUrlChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setUrl(value);
    if (value) {
      setSelectedFile(null);
    }
  }, []);

  const handleSubmit = useCallback(() => {
    if (selectedFile) {
      onSubmit({ file: selectedFile });
    } else if (url.trim()) {
      onSubmit({ url: url.trim() });
    }
  }, [selectedFile, url, onSubmit]);

  const handleFileButtonClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const hasInput = url.trim() || selectedFile;

  return (
    <div className="w-full max-w-3xl mx-auto">
      {/* Input Container - ChatGPT style */}
      <div
        className={`
          relative rounded-3xl transition-colors shadow-sm
          ${isDragging ? "border-2 border-emerald-400 bg-emerald-50/30" : "border border-gray-300 bg-white"}
          ${selectedFile ? "border-emerald-400" : ""}
        `}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex items-center p-4 pr-2">
          {/* File Upload Button (Left side) */}
          <button
            type="button"
            onClick={handleFileButtonClick}
            className="mr-3 p-2 rounded-lg hover:bg-gray-100 transition-colors flex-shrink-0"
            title="Upload PDF"
          >
            <Upload className="w-5 h-5 text-gray-600" />
          </button>

          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileChange}
            className="hidden"
          />

          {/* URL Input */}
          <input
            type="text"
            placeholder={selectedFile ? selectedFile.name : "Enter research paper URL or upload PDF..."}
            value={url}
            onChange={handleUrlChange}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && hasInput) {
                handleSubmit();
              }
            }}
            className="flex-1 outline-none text-gray-800 placeholder-gray-400 bg-transparent text-base py-2"
            disabled={!!selectedFile}
          />

          {/* Submit Button (Right side, circular) */}
          <button
            onClick={handleSubmit}
            disabled={!hasInput}
            className={`
              ml-3 p-2.5 rounded-full transition-colors flex-shrink-0
              ${
                hasInput
                  ? "bg-emerald-500 hover:bg-emerald-600 text-white shadow-sm"
                  : "bg-gray-200 text-gray-400 cursor-not-allowed"
              }
            `}
            title="Analyze"
          >
            <ArrowUp className="w-5 h-5" strokeWidth={2} />
          </button>
        </div>

        {/* File Selected Indicator */}
        {selectedFile && (
          <div className="px-4 pb-3">
            <div className="flex items-center justify-between bg-emerald-50 rounded-xl p-3 border border-emerald-200">
              <span className="text-sm text-emerald-800 truncate font-medium">
                📄 {selectedFile.name}
              </span>
              <button
                onClick={() => setSelectedFile(null)}
                className="ml-2 text-emerald-600 hover:text-emerald-800 transition-colors"
              >
                ✕
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Subtle hint text below input */}
      <p className="text-center text-xs text-gray-400 mt-3">
        Logos can analyze research papers from URLs or PDF files
      </p>
    </div>
  );
});

FileUploader.displayName = 'FileUploader';

export default FileUploader;
