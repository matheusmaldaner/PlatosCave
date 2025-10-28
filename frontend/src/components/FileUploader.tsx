// PlatosCave/frontend/src/components/FileUploader.tsx
import React, { useCallback, useState, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';

interface FileUploaderProps {
  onFileUpload: (file: File) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileUpload }) => {
<<<<<<< HEAD
    const [url, setUrl] = useState('');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const [isMultiline, setIsMultiline] = useState(false);

    // Minimal auto-resize to keep current layout logic working
    const autoResizeTextarea = useCallback(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;
        textarea.style.height = 'auto';
        const scrollHeight = textarea.scrollHeight;
        textarea.style.height = `${scrollHeight}px`;
        const computedStyle = window.getComputedStyle(textarea);
        const lineHeight = parseInt(computedStyle.lineHeight);
        setIsMultiline(scrollHeight > lineHeight * 1.35);
    }, []);
=======
  const [url, setUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isMultiline, setIsMultiline] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
>>>>>>> 85ef0925aace36ad256107910db1426ff72a764c

  // Setup a responsive check for multiline layout
  useEffect(() => {
    const updateLayout = () => setIsMultiline(window.innerWidth < 640);
    updateLayout();
    window.addEventListener('resize', updateLayout);
    return () => window.removeEventListener('resize', updateLayout);
  }, []);

  // Auto-resize for the textarea input
  const autoResizeTextarea = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  useEffect(() => {
    autoResizeTextarea();
  }, [url, selectedFile]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
    }
  }, []);

<<<<<<< HEAD
    // Rotating typewriter text for input placeholder overlay
    const titles = [
        'Attention Is All You Need',
        'BERT: Pre-training of Deep Bidirectional Transformers',
        'GPT-3: Language Models are Few-Shot Learners',
        'ResNet: Deep Residual Learning for Image Recognition',
        'Neural Ordinary Differential Equations',
        'Diffusion Models Beat GANs on Image Synthesis',
        'Playing Atari with Deep Reinforcement Learning',
        'U-Net: Convolutional Networks for Biomedical Image Segmentation'
    ];
    const [typedText, setTypedText] = useState('');
    const [titleIndex, setTitleIndex] = useState(0);
    const [charIndex, setCharIndex] = useState(0);
    const [isDeleting, setIsDeleting] = useState(false);
    // (Rotating noun removed; static title restored)

    useEffect(() => {
        const current = titles[titleIndex % titles.length];
        const typingSpeed = isDeleting ? 30 : 65;
        const pauseAtEnd = 1200;
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

    return (
        <div className="flex h-full w-full flex-col items-center justify-center px-4 py-12 sm:py-16">
            <div className="mx-auto w-full max-w-3xl space-y-12">
                {/* Hero Section */}
                <div className="space-y-6 text-center">
                    <h1 className="relative text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight tracking-tight animate-dissolve-up">
                        <span className="sleek-text subpixel-antialiased">Analyze research papers</span>
                    </h1>
                    <p className="mx-auto max-w-2xl text-base font-light leading-relaxed text-gray-600 sm:text-lg animate-dissolve-up-delayed">
                        Search by name, paste a URL, or upload a PDF to verify a research paper's integrity
                    </p>
                </div>
=======
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
    noClick: true,
  });
>>>>>>> 85ef0925aace36ad256107910db1426ff72a764c

  const handleSubmit = () => {
    if (selectedFile) {
      onFileUpload(selectedFile);
    } else if (url.trim()) {
      // To support URL submission, connect this to your backend logic where needed.
      console.log('Submitting URL:', url.trim());
      alert('URL submission is not yet implemented.');
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  // Display text for the main input area
  const getDisplayText = () => {
    if (selectedFile) return selectedFile.name;
    if (isDragActive) return 'Drop the PDF here...';
    return 'Search by paper name, paste URL, or upload PDF...';
  };

  return (
    <div className="flex h-full w-full flex-col items-center justify-center px-4 py-12 sm:py-16">
      <div className="mx-auto w-full max-w-3xl space-y-12">
        {/* Hero Section */}
        <div className="space-y-6 text-center">
          <h1 className="text-4xl font-semibold leading-tight tracking-tight text-gray-900 sm:text-5xl lg:text-6xl">
            Analyze research papers
          </h1>
          <p className="mx-auto max-w-2xl text-base font-light leading-relaxed text-gray-600 sm:text-lg">
            Search by name, paste a URL, or upload a PDF to verify a research paper's integrity
          </p>
        </div>

<<<<<<< HEAD
                {/* Footer Text */}
                <div className="space-y-3 text-center">
                    <p className="text-xs font-light uppercase tracking-widest text-gray-400">
                        A software created by the FINS group for the University of Florida AI Days Hackathon
                    </p>
                </div>
            </div>

            <p className="text-sm text-gray-400 mt-4">
                Plato's Cave uses deep research agents to turn papers into structured logic, then verify it.
            </p>
=======
        {/* Main Input Area */}
        <div {...getRootProps()} className="mx-auto w-full max-w-2xl">
          <div
            className={`relative flex w-full rounded-2xl border bg-white/85 px-4 py-4 text-left backdrop-blur-sm transition-all duration-300 ease-in-out ${
              isMultiline
                ? 'flex-col gap-3'
                : 'flex-col gap-4 sm:flex-row sm:items-center sm:gap-0 sm:px-5 sm:py-4'
            } ${
              isDragActive
                ? 'border-green-300 shadow-lg shadow-green-100/50 sm:scale-[1.01]'
                : 'border-gray-200 shadow-sm hover:border-gray-300 hover:shadow-md'
            }`}
          >
            {/* Hidden file input for react-dropzone */}
            <input {...getInputProps()} />
            {/* Hidden file input for manual select */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".pdf"
              className="hidden"
            />

            {/* Upload Icon Button (visible only when not multiline for layout) */}
            {!isMultiline && (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  fileInputRef.current?.click();
                }}
                className="flex h-11 w-11 items-center justify-center text-gray-400 transition-colors duration-200 hover:text-gray-600 sm:mr-3"
                aria-label="Upload PDF"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8-4-4m0 0L8 8m4-4v12"
                  />
                </svg>
              </button>
            )}

            {/* Text Input / Field */}
            <div
              className={`relative flex w-full items-center rounded-xl border border-transparent bg-white px-3 py-2 shadow-inner focus-within:border-brand-green/60 ${
                isMultiline ? '' : 'flex-1'
              }`}
            >
              <textarea
                ref={textareaRef}
                value={selectedFile ? '' : url}
                onChange={(e) => {
                  setUrl(e.target.value);
                  if (selectedFile) setSelectedFile(null);
                  autoResizeTextarea();
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && (selectedFile || url.trim())) {
                    e.preventDefault();
                    handleSubmit();
                  }
                }}
                placeholder={getDisplayText()}
                className="w-full resize-none border-none bg-transparent text-base text-gray-800 placeholder:text-gray-400 focus:outline-none min-h-[24px]"
                readOnly={!!selectedFile}
                rows={1}
              />
              {selectedFile && (
                <span
                  className="ml-3 hidden max-w-[200px] truncate text-sm font-medium text-gray-600 sm:inline"
                  title={selectedFile.name}
                >
                  {selectedFile.name}
                </span>
              )}
            </div>

            {/* Action Buttons */}
            {isMultiline ? (
              <div className="flex items-center justify-between">
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  className="flex h-11 w-11 items-center justify-center text-gray-400 transition-colors duration-200 hover:text-gray-600"
                  aria-label="Upload PDF"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8-4-4m0 0L8 8m4-4v12"
                    />
                  </svg>
                </button>
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    handleSubmit();
                  }}
                  disabled={!selectedFile && !url.trim()}
                  className={`flex h-11 w-11 items-center justify-center rounded-full transition-colors duration-200 ${
                    selectedFile || url.trim()
                      ? 'bg-green-50 text-green-600 hover:bg-green-100'
                      : 'cursor-not-allowed bg-gray-100 text-gray-300'
                  }`}
                  aria-label="Analyze"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z" />
                  </svg>
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  handleSubmit();
                }}
                disabled={!selectedFile && !url.trim()}
                className={`flex h-11 w-11 items-center justify-center rounded-full transition-colors duration-200 sm:ml-3 ${
                  selectedFile || url.trim()
                    ? 'bg-green-50 text-green-600 hover:bg-green-100'
                    : 'cursor-not-allowed bg-gray-100 text-gray-300'
                }`}
                aria-label="Analyze"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z" />
                </svg>
              </button>
            )}
          </div>
>>>>>>> 85ef0925aace36ad256107910db1426ff72a764c
        </div>

        {/* Footer Text */}
        <div className="space-y-3 text-center">
          <p>
            A software created by the FINS group for the University of Florida AI Days Hackathon
          </p>
        </div>
      </div>

      <p className="text-sm text-gray-400 mt-4">
        Platos-Cave can analyze research papers from URLs or PDF files
      </p>
    </div>
  );
};

export default FileUploader;
