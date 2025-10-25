// frontend/src/components/ProgressBar.tsx
import React from 'react';
import { ProcessStep } from './Sidebar'; // Reusing this type definition

interface ProgressBarProps {
    steps: ProcessStep[];
}

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
        <div className="w-full max-w-3xl mx-auto my-4">
            <div className="flex items-center">
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
                            {/* Circle */}
                            <div className={circleClasses.join(' ')}>
                                {getStepIcon(step.status, index)}
                            </div>
                            {/* Line */}
                            {!isLast && (
                                <div
                                    className={[
                                        'flex-1 h-1 rounded-full transition-all duration-500',
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
                <p className="text-center text-gray-500 mt-2">
                    Step {activeIndex + 1} of {steps.length}: {activeStep.displayText}
                </p>
            )}
        </div>
    );
};

export default ProgressBar;