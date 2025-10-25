// PlatosCave/frontend/src/components/Sidebar.tsx
import React, { useEffect, useRef } from 'react';

// Define the shape of a process step object
export interface ProcessStep {
    name: string;
    displayText: string;
    status: 'pending' | 'active' | 'completed';
}

interface SidebarProps {
    steps: ProcessStep[];
    finalScore: number | null;
}

const Sidebar: React.FC<SidebarProps> = ({ steps, finalScore }) => {
    const activeStepRef = useRef<HTMLDivElement>(null);

    // This effect scrolls the active item into the center of the view
    useEffect(() => {
        if (activeStepRef.current) {
            activeStepRef.current.scrollIntoView({
                behavior: 'smooth',
                block: 'center',
            });
        }
    }, [steps.find(s => s.status === 'active')?.name]); // Scroll when the active step's name changes

    return (
        <div className="flex-grow overflow-y-auto flex flex-col items-center py-10 w-full">
            {steps.map((step, index) => (
                <React.Fragment key={step.name}>
                    <div
                        ref={step.status === 'active' ? activeStepRef : null}
                        className={`w-4/5 p-3 my-2 rounded-md text-center text-text-primary shadow-sm transition-all duration-300
                            ${step.status === 'active' ? 'bg-brand-green-light scale-105' : 'bg-gray-200'}
                            ${step.status === 'completed' ? 'bg-brand-green-light opacity-70' : ''}`}
                    >
                        <div className="font-bold text-md">{step.name}</div>
                        <div className="text-sm text-text-secondary">{step.displayText}</div>
                    </div>

                    {/* Render arrow unless it's the last step */}
                    {index < steps.length - 1 && (
                         <div className="text-gray-300 text-2xl font-light">↓</div>
                    )}
                </React.Fragment>
            ))}

            {/* Render the final score when it's available */}
            {finalScore !== null && (
                <>
                    <div className="text-gray-300 text-2xl font-light">↓</div>
                    <div
                        className="w-4/5 p-3 my-2 rounded-md text-center text-white font-bold bg-brand-green shadow-lg"
                    >
                        {finalScore}
                    </div>
                </>
            )}
        </div>
    );
};

export default Sidebar;