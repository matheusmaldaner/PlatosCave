// PlatosCave/frontend/src/components/landing/PaperCard.tsx
import React from 'react';
import { Bell, BookOpen, CheckCircle2, Check, Lock } from 'lucide-react';

interface PaperCardProps {
  onViewAnalysis?: () => void;
}

const PaperCard: React.FC<PaperCardProps> = ({ onViewAnalysis }) => {
  const checks = [
    { label: 'Citation accuracy verified', status: 'pass' },
    { label: 'Methodology reproducible', status: 'pass' },
    { label: 'Statistical analysis sound', status: 'pass' },
    { label: 'No plagiarism detected', status: 'pass' },
  ];

  return (
    <div className="relative animate-fade-in-right">
      {/* Alert Badge - Floating */}
      <div className="absolute -right-2 -top-3 z-10 flex items-center gap-2 rounded-xl border border-gray-100 bg-white px-3 py-2 shadow-card-lg animate-float">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-orange-50">
          <Bell className="h-4 w-4 text-orange-500" />
        </div>
        <div className="text-left">
          <p className="text-xs font-semibold text-text-primary">New Alert</p>
          <p className="text-[10px] text-brand-green">3 citations flagged</p>
        </div>
      </div>

      {/* Main Card */}
      <div className="overflow-hidden rounded-3xl border border-gray-100 bg-white shadow-card-xl">
        {/* Card Header */}
        <div className="border-b border-gray-100 px-6 py-5">
          <div className="flex items-start justify-between gap-4">
            {/* Institution Info */}
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gray-100">
                <BookOpen className="h-5 w-5 text-gray-600" strokeWidth={1.5} />
              </div>
              <div>
                <p className="text-sm font-semibold text-text-primary">OpenAI Research</p>
                <p className="text-xs text-text-muted">Posted 2 hours ago • AI/ML</p>
              </div>
            </div>
            
            {/* Match Badge */}
            <div className="flex items-center gap-1.5 rounded-full border border-brand-green/20 bg-brand-green-50 px-3 py-1.5">
              <CheckCircle2 className="h-3.5 w-3.5 text-brand-green" />
              <span className="text-xs font-semibold text-brand-green">92% Integrity</span>
            </div>
          </div>
        </div>

        {/* Card Body */}
        <div className="px-6 py-5">
          {/* Paper Title */}
          <h3 className="mb-4 font-serif text-xl font-semibold leading-tight text-text-primary">
            GPT-3: Language Models are Few-Shot Learners
          </h3>

          {/* Tags */}
          <div className="mb-5 flex flex-wrap gap-2">
            <span className="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-text-secondary">
              ArXiv 2020
            </span>
            <span className="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-text-secondary">
              NLP
            </span>
            <span className="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-text-secondary">
              Deep Learning
            </span>
          </div>

          {/* Verification Checks */}
          <div className="space-y-3">
            {checks.map((check, index) => (
              <div key={index} className="flex items-center gap-3">
                <div className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-green-50">
                  <Check className="h-3 w-3 text-brand-green" strokeWidth={3} />
                </div>
                <span className="text-sm text-text-secondary">{check.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Card Footer */}
        <div className="px-6 pb-6">
          <button
            onClick={onViewAnalysis}
            className="group flex w-full items-center justify-center gap-2 rounded-xl bg-text-primary py-3.5 text-sm font-medium text-white transition-all duration-200 hover:bg-gray-800 hover:shadow-lg active:scale-[0.99]"
          >
            View full analysis
            <Lock className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-0.5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default PaperCard;
