// frontend/src/components/ProgressBar.tsx
import React from "react";
import { ProcessStep } from "../types";

interface ProgressBarProps {
    steps: ProcessStep[];
}

// Tooltip descriptions for each stage
const STAGE_DESCRIPTIONS: Record<string, string> = {
    "Validate": "URL validation & browser setup",
    "Decomposing PDF": "Extract paper content",
    "Building Logic Tree": "Identify claims & evidence",
    "Organizing Agents": "Deploy verification agents",
    "Compiling Evidence": "Verify all claims",
    "Evaluating Integrity": "Calculate final score"
};

const ACCENT = "#1f7898"; // Academic Minimalist teal

const getStepIcon = (status: "pending" | "active" | "completed", index: number) => {
  if (status === "completed") {
    return (
      <svg
        xmlns="http://www.w3.org/2000/svg"
        className="h-4 w-4"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        style={{ color: "white" }}
        aria-hidden="true"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
      </svg>
    );
  }

  if (status === "active") {
    return <span className="h-2 w-2 rounded-full" style={{ background: ACCENT }} aria-hidden="true" />;
  }

  return (
    <span className="text-[11px] font-semibold tabular-nums text-gray-500" aria-hidden="true">
      {index + 1}
    </span>
  );
};

const ProgressBar: React.FC<ProgressBarProps> = ({ steps }) => {
    const activeStep = steps.find(step => step.status === 'active');
    const activeIndex = activeStep ? steps.indexOf(activeStep) : -1;

    return (
      <div className="mx-auto w-full max-w-4xl px-4 pb-2 pt-3 md:px-6">
        <div className="flex items-center gap-2 overflow-x-auto pb-2 md:overflow-visible md:pb-0">
          {steps.map((step, index) => {
            const isCompleted = step.status === "completed";
            const isActive = step.status === "active";
            const isLast = index === steps.length - 1;

            const circleBase =
              "w-7 h-7 rounded-full flex items-center justify-center z-10 transition-colors duration-200";

            const circleStyle: React.CSSProperties = isCompleted
              ? { background: ACCENT, color: "white", boxShadow: "0 6px 14px rgba(0,0,0,0.06)" }
              : isActive
              ? { background: "white", color: ACCENT, boxShadow: "0 6px 14px rgba(0,0,0,0.06)", border: `1px solid ${ACCENT}` }
              : { background: "#ecebea", color: "#6b7280", border: "1px solid #e5e7eb" };

            const lineStyle: React.CSSProperties = isCompleted
              ? { background: ACCENT, opacity: 0.7 }
              : isActive
              ? { background: ACCENT, opacity: 0.28 }
              : { background: "#e5e7eb" };

            return (
              <React.Fragment key={step.name}>
                <div className="relative group cursor-help">
                  <div className={circleBase} style={circleStyle}>
                    {getStepIcon(step.status as any, index)}
                  </div>

                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2.5 py-1.5 bg-gray-900 text-white text-xs rounded-md opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 pointer-events-none whitespace-nowrap z-[9999] shadow-lg">
                    <div className="text-gray-300">
                      {index + 1}/{steps.length}: {STAGE_DESCRIPTIONS[step.name] || step.name}
                    </div>
                    <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px">
                      <div className="border-[3px] border-transparent border-t-gray-900"></div>
                    </div>
                  </div>
                </div>

                {!isLast && (
                  <div
                    className="flex-1 h-[2px] min-w-[40px] rounded-full transition-all duration-300"
                    style={lineStyle}
                  />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {activeStep && (
          <p className="mt-1.5 text-center text-[12px] text-gray-500">
            <span className="font-medium text-gray-700">
              Step {activeIndex + 1} of {steps.length}
            </span>
            <span className="text-gray-400"> — </span>
            <span className="tracking-wide">{activeStep.displayText}</span>
          </p>
        )}
      </div>
    );
};

export default ProgressBar;