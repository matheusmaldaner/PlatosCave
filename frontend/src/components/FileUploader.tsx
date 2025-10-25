// PlatosCave/frontend/src/components/FileUploader.tsx
import React, { useCallback, useState, useRef } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploaderProps {
    onFileUpload: (file: File) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload }) => {
    const [url, setUrl] = useState('');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            setSelectedFile(acceptedFiles[0]);
            // For immediate upload on drop, uncomment the next line
            // onFileUpload(acceptedFiles[0]);
        }
    }, [onFileUpload]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'application/pdf': ['.pdf'] },
        multiple: false,
        noClick: true, // We will trigger the click manually
    });

    const handleSubmit = () => {
        if (selectedFile) {
            onFileUpload(selectedFile);
        } else if (url) {
            // Future logic to handle URL submission
            console.log('Submitting URL:', url);
            alert("URL submission is not yet implemented.");
        }
    };

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setSelectedFile(event.target.files[0]);
        }
    };
    
    // Get the display text for the input
    const getDisplayText = () => {
        if (selectedFile) {
            return selectedFile.name;
        }
        if (isDragActive) {
            return "Drop the PDF here...";
        }
        return "Enter research paper URL or upload PDF...";
    };

    return (
        <div className="flex flex-col items-center justify-center h-full w-full text-center">
            <h1 className="text-4xl font-bold text-text-primary mb-2">
                Analyze research papers instantly
            </h1>
            <p className="text-lg text-text-secondary mb-8">
                Upload a PDF or paste a URL to extract insights from research papers
            </p>

            {/* Main Input Area */}
            <div {...getRootProps()} className="w-full max-w-2xl mx-auto">
                <div className={`relative flex items-center w-full p-2 bg-white rounded-xl shadow-md border transition-all ${isDragActive ? 'border-brand-green ring-2 ring-brand-green-light' : 'border-gray-200'}`}>
                    {/* Hidden file input for react-dropzone */}
                    <input {...getInputProps()} />
                    {/* Hidden file input for manual selection */}
                    <input type="file" ref={fileInputRef} onChange={handleFileSelect} accept=".pdf" style={{ display: 'none' }} />

                    {/* Upload Icon */}
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        className="p-2 text-gray-400 hover:text-gray-600"
                        aria-label="Upload PDF"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                    </button>

                    {/* Text Input / File Name Display */}
                    <input
                        type="text"
                        value={selectedFile ? '' : url} // Clear URL input if a file is selected
                        onChange={(e) => {
                            setUrl(e.target.value);
                            if (selectedFile) setSelectedFile(null); // Clear file if user starts typing
                        }}
                        placeholder={getDisplayText()}
                        className="flex-grow bg-transparent outline-none border-none text-gray-700 mx-2 placeholder:text-gray-400"
                        readOnly={!!selectedFile} // Make input readonly if a file is selected
                    />

                    {/* Submit Button */}
                    <button
                        onClick={handleSubmit}
                        disabled={!selectedFile && !url}
                        className="w-8 h-8 flex items-center justify-center bg-gray-200 rounded-lg transition hover:bg-gray-300 disabled:bg-gray-100 disabled:cursor-not-allowed"
                        aria-label="Submit"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 10l7-7m0 0l7 7m-7-7v18" />
                        </svg>
                    </button>
                </div>
            </div>

            <p className="text-sm text-gray-400 mt-4">
                Logos can analyze research papers from URLs or PDF files
            </p>
        </div>
    );
};

export default FileUploader;