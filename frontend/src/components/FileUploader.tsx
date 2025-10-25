// PlatosCave/frontend/src/components/FileUploader.tsx
import React, { useCallback, useState, useRef } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploaderProps {
    onFileUpload: (file: File) => void;
    onUrlSubmit: (url: string) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload, onUrlSubmit }) => {
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
        } else if (url.trim()) {
            onUrlSubmit(url.trim());
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
        <div className="flex flex-col items-center justify-center h-full w-full text-center px-4">
            <div className="max-w-3xl mx-auto space-y-12">
                {/* Hero Section */}
                <div className="space-y-6">
                    <h1 className="text-6xl font-semibold text-gray-900 tracking-tight leading-tight">
                        Analyze research papers
                    </h1>
                    <p className="text-xl text-gray-600 font-light max-w-2xl mx-auto leading-relaxed">
                        Upload a PDF or enter a URL to extract insights and evaluate research integrity
                    </p>
                </div>

                {/* Main Input Area */}
                <div {...getRootProps()} className="w-full max-w-2xl mx-auto">
                    <div className={`relative flex items-center w-full px-5 py-4 bg-white/80 backdrop-blur-sm rounded-2xl border transition-all duration-300 ${isDragActive ? 'border-green-300 shadow-lg shadow-green-100/50 scale-[1.02]' : 'border-gray-200 shadow-sm hover:shadow-md hover:border-gray-300'}`}>
                        {/* Hidden file input for react-dropzone */}
                        <input {...getInputProps()} />
                        {/* Hidden file input for manual selection */}
                        <input type="file" ref={fileInputRef} onChange={handleFileSelect} accept=".pdf" style={{ display: 'none' }} />

                        {/* Upload Icon */}
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            className="p-2 text-gray-400 hover:text-gray-700 transition-colors duration-200"
                            aria-label="Upload PDF"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                            </svg>
                        </button>

                        {/* Text Input / File Name Display */}
                        <input
                            type="text"
                            value={selectedFile ? '' : url}
                            onChange={(e) => {
                                setUrl(e.target.value);
                                if (selectedFile) setSelectedFile(null);
                            }}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && (selectedFile || url.trim())) {
                                    handleSubmit();
                                }
                            }}
                            placeholder={getDisplayText()}
                            className="flex-grow bg-transparent outline-none border-none text-gray-800 text-base mx-3 placeholder:text-gray-400"
                            readOnly={!!selectedFile}
                        />

                        {/* Submit Button */}
                        <button
                            onClick={handleSubmit}
                            disabled={!selectedFile && !url}
                            className={`px-6 py-2 rounded-xl font-medium transition-all duration-200 ${
                                selectedFile || url
                                    ? 'bg-gradient-to-r from-green-400 to-green-500 text-white hover:from-green-500 hover:to-green-600 shadow-sm hover:shadow-md'
                                    : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                            }`}
                            aria-label="Submit"
                        >
                            Analyze
                        </button>
                    </div>
                </div>

                {/* Footer Text */}
                <div className="space-y-3">
                    <p className="text-sm text-gray-500 font-light">
                        A software created by the FINS group for the University of Florida AI Days Hackathon
                    </p>
                </div>
            </div>
        </div>
    );
};

export default FileUploader;