// Minimal PDF -> text extraction using PDF.js via CDN UMD build to avoid bundler issues
// No local dependency required; we dynamically load scripts in the browser.

declare global {
  interface Window {
    pdfjsLib?: any;
  }
}

let pdfjsReadyPromise: Promise<any> | null = null;

async function ensurePdfJsLoaded(): Promise<any> {
  if (typeof window === "undefined") throw new Error("PDF.js must run in the browser");
  if (window.pdfjsLib) return window.pdfjsLib;
  if (!pdfjsReadyPromise) {
    pdfjsReadyPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://unpkg.com/pdfjs-dist@3.11.174/legacy/build/pdf.min.js";
      script.async = true;
      script.onload = () => {
        try {
          const lib = window.pdfjsLib;
          if (!lib) throw new Error("pdfjsLib not found on window");
          lib.GlobalWorkerOptions.workerSrc = "https://unpkg.com/pdfjs-dist@3.11.174/legacy/build/pdf.worker.min.js";
          resolve(lib);
        } catch (e) {
          reject(e);
        }
      };
      script.onerror = () => reject(new Error("Failed to load PDF.js"));
      document.head.appendChild(script);
    });
  }
  return pdfjsReadyPromise;
}

export async function extractPdfText(file: File): Promise<string> {
  const startedAt = Date.now();
  const buf = await file.arrayBuffer();
  const pdfjs = await ensurePdfJsLoaded();
  const loadingTask = pdfjs.getDocument({ data: buf });
  const pdf = await loadingTask.promise;
  const numPages: number = pdf.numPages;
  console.log("PDF.js loaded", { pages: numPages, ms: Date.now() - startedAt });

  const pageTexts: string[] = [];
  for (let i = 1; i <= numPages; i++) {
    const page = await pdf.getPage(i);
    const tc = await page.getTextContent();
    const text = (tc.items as any[])
      .map((it) => (typeof (it as any).str === "string" ? (it as any).str : ""))
      .filter(Boolean)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    pageTexts.push(text);
  }
  return pageTexts.join("\n\n");
}


