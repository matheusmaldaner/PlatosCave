// frontend/src/components/ProgressBar.tsx
import React from 'react';
import { ProcessStep } from '../types';

interface ProgressBarProps {
    steps: ProcessStep[];
}

// Tooltip descriptions for each stage
const STAGE_DESCRIPTIONS: Record<string, string> = {
    "Validate": "URL validation & browser setup",
    "Decomposing PDF": "Extract paper content",
    "Building Logic Tree": "Identify claims & evidence",
    "Building Knowledge Graph": "Identify claims & evidence",
    "Organizing Agents": "Deploy verification agents",
    "Compiling Evidence": "Verify all claims",
    "Evaluating Integrity": "Calculate final score"
};

const getStepIcon = (status: 'pending' | 'active' | 'completed', index: number) => {
    if (status === 'completed') {
        return (
            <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
            >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
            </svg>
        );
    }

    if (status === 'active') {
        return (
            <div className="relative flex items-center justify-center">
                <span className="absolute inline-flex h-5 w-5 rounded-full bg-brand-green/30 opacity-70 animate-ping" />
                <span className="absolute inline-flex h-3 w-3 rounded-full bg-brand-green/70 animate-pulse" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-brand-green" />
            </div>
        );
    }

    return <span className="text-gray-500 font-semibold">{index + 1}</span>;
};

const ProgressBar: React.FC<ProgressBarProps> = ({ steps }) => {
    const activeStep = steps.find(step => step.status === 'active');
    const activeIndex = activeStep ? steps.indexOf(activeStep) : -1;

    return (
        <div className="mx-auto my-4 w-full max-w-4xl px-4 md:px-6">
            <div className="flex items-center gap-2 overflow-x-auto pb-2 md:overflow-visible md:pb-0">
                {steps.map((step, index) => {
                    const isCompleted = step.status === 'completed';
                    const isActive = step.status === 'active';
                    const isLast = index === steps.length - 1;

                    const circleClasses = [
                        'w-8 h-8 rounded-full flex items-center justify-center z-10 transition-all duration-300'
                    ];

                    if (isCompleted) {
                        circleClasses.push('bg-brand-green shadow-sm text-white');
                    } else if (isActive) {
                        circleClasses.push('bg-white ring-2 ring-brand-green/70 text-brand-green');
                    } else {
                        circleClasses.push('bg-gray-200 text-gray-500');
                    }

                    return (
                        <React.Fragment key={step.name}>
                            {/* Circle with Tooltip */}
                            <div className="relative group cursor-help">
                                <div className={`${circleClasses.join(' ')} group-hover:scale-110`}>
                                    {getStepIcon(step.status, index)}
                                </div>
                                {/* Tooltip */}
                                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2.5 py-1.5 bg-gray-900 text-white text-xs rounded-md opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 pointer-events-none whitespace-nowrap z-[9999] shadow-lg">
                                    <div className="text-gray-300">
                                        {index + 1}/{steps.length}: {STAGE_DESCRIPTIONS[step.name] || step.name}
                                    </div>
                                    {/* Arrow pointing down */}
                                    <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px">
                                        <div className="border-[3px] border-transparent border-t-gray-900"></div>
                                    </div>
                                </div>
                            </div>
                            {/* Line */}
                            {!isLast && (
                                <div
                                    className={[
                                        'flex-1 h-1 min-w-[40px] rounded-full transition-all duration-500',
                                        isCompleted
                                            ? 'bg-brand-green'
                                            : isActive
                                            ? 'bg-brand-green/40 animate-pulse'
                                            : 'bg-gray-200'
                                    ].join(' ')}
                                />
                            )}
                        </React.Fragment>
                    );
                })}
            </div>
            {/* Step Text */}
            {activeStep && (
                <p className="mt-2 text-center text-sm text-gray-500 md:text-base">
                    Step {activeIndex + 1} of {steps.length}: {activeStep.displayText}
                </p>
            )}
        </div>
    );
};

export default ProgressBar;