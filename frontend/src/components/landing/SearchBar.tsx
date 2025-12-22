// PlatosCave/frontend/src/components/landing/SearchBar.tsx
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, X, Play } from 'lucide-react';

interface SearchBarProps {
  onFileUpload: (file: File) => void;
  onUrlSubmit: (url: string) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ onFileUpload, onUrlSubmit }) => {
  const [url, setUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Typewriter effect for placeholder
  const titles = [
    'Attention Is All You Need',
    'BERT: Pre-training of Deep Bidirectional Transformers',
    'GPT-3: Language Models are Few-Shot Learners',
    'Neural Ordinary Differential Equations',
  ];

  const [typedText, setTypedText] = useState('');
  const [titleIndex, setTitleIndex] = useState(0);
  const [charIndex, setCharIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    const current = titles[titleIndex % titles.length];
    const typingSpeed = isDeleting ? 25 : 55;
    const pauseAtEnd = 1500;

    let timer: number;

    if (!isDeleting && charIndex < current.length) {
      timer = window.setTimeout(() => {
        setTypedText(current.slice(0, charIndex + 1));
        setCharIndex(charIndex + 1);
      }, typingSpeed);
    } else if (!isDeleting && charIndex === current.length) {
      timer = window.setTimeout(() => setIsDeleting(true), pauseAtEnd);
    } else if (isDeleting && charIndex > 0) {
      timer = window.setTimeout(() => {
        setTypedText(current.slice(0, charIndex - 1));
        setCharIndex(charIndex - 1);
      }, typingSpeed);
    } else if (isDeleting && charIndex === 0) {
      setIsDeleting(false);
      setTitleIndex((titleIndex + 1) % titles.length);
    }

    return () => window.clearTimeout(timer);
  }, [charIndex, isDeleting, titleIndex, titles]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
    noClick: true,
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

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (selectedFile || url.trim())) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setUrl('');
  };

  return (
    <div {...getRootProps()} className="w-full animate-fade-in-up-delayed-2">
      {/* Hidden inputs */}
      <input {...getInputProps()} />
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileSelect} 
        accept=".pdf" 
        className="hidden" 
      />

      {/* Search Container */}
      <div
        className={`relative flex items-center gap-3 rounded-full bg-white px-4 py-3 shadow-search transition-all duration-300 ${
          isDragActive
            ? 'ring-2 ring-brand-green ring-offset-2'
            : 'hover:shadow-lg'
        }`}
      >
        {/* Upload Button */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
          className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-gray-100 text-text-secondary transition-all duration-200 hover:bg-gray-200 hover:text-text-primary"
          aria-label="Upload PDF"
        >
          <Upload className="h-5 w-5" />
        </button>

        {/* Input Field */}
        <div className="relative flex-1">
          {selectedFile ? (
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-brand-green" />
              <span className="truncate text-sm font-medium text-text-primary">
                {selectedFile.name}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  clearSelection();
                }}
                className="ml-1 rounded-full p-0.5 text-text-muted hover:bg-gray-100 hover:text-text-primary"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder=""
                className="w-full bg-transparent text-sm text-text-primary placeholder:text-text-muted focus:outline-none"
              />
              {url.trim() === '' && (
                <span className="pointer-events-none absolute inset-0 flex items-center text-sm text-text-muted typing-caret">
                  {typedText || 'Search by title, DOI, URL, or drop a PDF…'}
                </span>
              )}
            </>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            handleSubmit();
          }}
          disabled={!selectedFile && !url.trim()}
          className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full transition-all duration-200 ${
            selectedFile || url.trim()
              ? 'bg-text-primary text-white hover:bg-gray-800 hover:shadow-lg active:scale-95'
              : 'cursor-not-allowed bg-gray-100 text-text-muted'
          }`}
          aria-label="Analyze"
        >
          <Play className="h-5 w-5" fill="currentColor" />
        </button>
      </div>

      {/* Drag overlay text */}
      {isDragActive && (
        <div className="mt-3 text-center text-sm font-medium text-brand-green">
          Drop your PDF here to analyze
        </div>
      )}
    </div>
  );
};

export default SearchBar;
