// frontend/src/components/ProgressBar.tsx
import React from 'react';
import { ProcessStep } from './Sidebar'; // Reusing this type definition

interface ProgressBarProps {
    steps: ProcessStep[];
}

const getStepIcon = (status: 'pending' | 'active' | 'completed', index: number) => {
    // Completed Icon (Green Checkmark)
    if (status === 'completed') {
        return <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>;
    }
    // Active Icon (Green Spinner)
    if (status === 'active') {
        return <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>;
    }
    // Pending Icon (Gray Number)
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

                    return (
                        <React.Fragment key={step.name}>
                            {/* Circle */}
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center z-10 transition-all duration-300
                                ${isCompleted ? 'bg-brand-green' : ''}
                                ${isActive ? 'bg-brand-green' : ''} {/* CHANGED: from dark green to regular green */}
                                ${step.status === 'pending' ? 'bg-gray-300' : ''} {/* This handles the "white until complete" request */}
                            `}>
                                {getStepIcon(step.status, index)}
                            </div>
                            {/* Line */}
                            {!isLast && (
                                <div className={`flex-1 h-1 transition-all duration-500
                                    ${index < activeIndex || isCompleted ? 'bg-brand-green' : ''} {/* Completed lines are also green */}
                                    ${index === activeIndex ? 'bg-brand-green' : ''} {/* CHANGED: from dark green to regular green */}
                                    ${step.status === 'pending' && index > activeIndex ? 'bg-gray-300' : ''}
                                `}></div>
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