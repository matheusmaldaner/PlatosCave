import React from "react";
import type { HeadFC, PageProps } from "gatsby";
import FileUploader from "../components/FileUploader";
import { extractFactDAGWithLLM, type FactDAG } from "../utils/factDagLLM";
import { ApiDagLLM } from "../utils/factDagClient";

const IndexPage: React.FC<PageProps> = () => {
  const handleSubmit = async (data: { url?: string; file?: File }) => {
    if (data.url) {
      console.log("📄 Analyzing URL:", data.url);
    } else if (data.file) {
      console.log("📄 Analyzing PDF:", data.file.name);
      console.log("   File size:", (data.file.size / 1024).toFixed(2), "KB");
      console.log("   File type:", data.file.type);

      // Convert PDF to text (stubbed for now). Replace with real PDF.js extraction.
      const extractedText = await fakeExtractPdfText(data.file);

      // LLM-driven single global DAG of facts (via backend API)
      const dagLlm = new ApiDagLLM();
      const factDag: FactDAG = await extractFactDAGWithLLM(extractedText, dagLlm);
      console.log("🔗 Fact DAG:", { nodes: factDag.nodes.length, edges: factDag.edges.length });
      console.log(factDag.nodes.slice(0, 3));
      console.log(factDag.edges.slice(0, 5));
    }
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Top Navigation Bar */}
      <nav className="fixed top-0 left-0 right-0 bg-white border-b border-gray-200 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          {/* Logo on the left */}
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">L</span>
            </div>
            <span className="font-semibold text-gray-800 text-lg">Logos</span>
          </div>

          {/* Right side controls (placeholder for future additions) */}
          <div className="flex items-center space-x-3">
            {/* Add menu/profile icons here if needed */}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 pt-16">
        {/* Center Message */}
        <div className="text-center mb-8 max-w-2xl">
          <h1 className="text-3xl font-medium text-gray-800 mb-3">
            Analyze research papers instantly
          </h1>
          <p className="text-gray-500">
            Upload a PDF or paste a URL to extract insights from research papers
          </p>
        </div>

        {/* File Uploader Component */}
        <div className="w-full max-w-3xl">
          <FileUploader onSubmit={handleSubmit} />
        </div>
      </main>
    </div>
  );
};

export default IndexPage;

export const Head: HeadFC = () => (
  <>
    <title>Logos - Research Paper Summarizer</title>
    <meta name="description" content="Analyze research papers from URLs or PDFs" />
  </>
);

// Temporary PDF-to-text stub. Replace with actual PDF parsing (e.g., PDF.js) later.
async function fakeExtractPdfText(file: File): Promise<string> {
  // In production, use a real extractor and return the full text content
  return `Title: Example Paper\n\nThis is a sample paragraph introducing the topic. It contains several sentences.\n\nMethods: We apply a straightforward approach. Results show improvements of 12%.\n\nConclusion: The proposed method is effective.`;
}
