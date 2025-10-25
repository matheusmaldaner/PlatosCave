// PlatosCave/frontend/src/components/FileUploader.tsx
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploaderProps {
    onFileUpload: (file: File) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload }) => {
    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            onFileUpload(acceptedFiles[0]);
        }
    }, [onFileUpload]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'application/pdf': ['.pdf'] },
        multiple: false,
    });

    return (
        <div className="flex items-center justify-center h-full w-full">
            <div
                {...getRootProps()}
                className="w-full max-w-lg h-48 border-2 border-dashed border-brand-green-light rounded-lg flex flex-col items-center justify-center cursor-pointer transition hover:border-brand-green bg-base-gray"
            >
                <input {...getInputProps()} />
                <svg className="w-12 h-12 text-brand-green mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M12 15l-3-3m0 0l3-3m-3 3h12"></path></svg>
                {isDragActive ? (
                    <p className="text-text-primary">Drop the PDF here ...</p>
                ) : (
                    <p className="text-text-primary">
                        Drag & drop a PDF here, or click to select a file
                    </p>
                )}
            </div>
        </div>
    );
};

export default FileUploader;