// PlatosCave/frontend/src/components/landing/Hero.tsx
import React from 'react';
import { ArrowDown, ArrowRight } from 'lucide-react';
import SearchBar from './SearchBar';
import PaperCard from './PaperCard';

interface HeroProps {
  onFileUpload: (file: File) => void;
  onUrlSubmit: (url: string) => void;
  onViewExamples?: () => void;
}

const Hero: React.FC<HeroProps> = ({ onFileUpload, onUrlSubmit, onViewExamples }) => {
  const trustedAvatars = [
    { initials: 'UF', color: 'bg-blue-100 text-blue-600' },
    { initials: 'MIT', color: 'bg-purple-100 text-purple-600' },
    { initials: 'SU', color: 'bg-green-100 text-green-600' },
  ];

  return (
    <section className="relative overflow-hidden px-6 py-16 lg:py-24">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-16 lg:items-center">
          {/* Left Column - Content */}
          <div className="space-y-8">
            {/* Hero Headline */}
            <div className="space-y-4 animate-fade-in-up-delayed">
              <h1 className="font-serif text-hero leading-tight text-text-primary lg:text-hero-lg">
                <span className="block font-semibold">Analyze research papers.</span>
                <span className="block font-normal italic text-text-secondary">
                  Verify integrity instantly.
                </span>
              </h1>
            </div>

            {/* Supporting Text */}
            <p className="max-w-lg text-lg leading-relaxed text-text-secondary animate-fade-in-up-delayed-2">
              Our AI-powered pipeline decomposes academic papers, builds knowledge graphs, 
              and evaluates citations, methodology, and reproducibility in minutes.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-wrap items-center gap-4 animate-fade-in-up-delayed-3">
              <button
                onClick={() => {
                  const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement;
                  searchInput?.focus();
                }}
                className="group inline-flex items-center gap-2 rounded-full bg-text-primary px-6 py-3.5 text-sm font-medium text-white transition-all duration-200 hover:bg-gray-800 hover:shadow-xl active:scale-[0.98]"
              >
                Start analyzing
                <ArrowDown className="h-4 w-4 transition-transform duration-200 group-hover:translate-y-0.5" />
              </button>
              <button
                onClick={onViewExamples}
                className="group inline-flex items-center gap-2 rounded-full border border-gray-300 bg-white px-6 py-3.5 text-sm font-medium text-text-primary transition-all duration-200 hover:border-gray-400 hover:bg-gray-50 active:scale-[0.98]"
              >
                View examples
                <ArrowRight className="h-4 w-4 transition-transform duration-200 group-hover:translate-x-0.5" />
              </button>
            </div>

            {/* Search Bar */}
            <div className="pt-4">
              <SearchBar onFileUpload={onFileUpload} onUrlSubmit={onUrlSubmit} />
            </div>

            {/* Trust Indicators */}
            <div className="flex items-center gap-4 pt-4 animate-fade-in-up-delayed-3">
              <div className="flex -space-x-2">
                {trustedAvatars.map((avatar, index) => (
                  <div
                    key={index}
                    className={`flex h-8 w-8 items-center justify-center rounded-full border-2 border-white text-[10px] font-semibold ${avatar.color}`}
                  >
                    {avatar.initials}
                  </div>
                ))}
              </div>
              <span className="text-sm text-text-muted">
                Trusted by researchers & educators
              </span>
            </div>
          </div>

          {/* Right Column - Paper Card */}
          <div className="relative lg:pl-8">
            {/* Decorative gradient blob */}
            <div className="absolute -right-20 -top-20 h-72 w-72 rounded-full bg-gradient-to-br from-brand-green-100/40 to-brand-green-200/30 blur-3xl"></div>
            <div className="absolute -bottom-10 -left-10 h-48 w-48 rounded-full bg-gradient-to-tr from-cream-300/50 to-cream-200/40 blur-2xl"></div>
            
            <div className="relative">
              <PaperCard 
                onViewAnalysis={() => {
                  const searchInput = document.querySelector('input[type="text"]') as HTMLInputElement;
                  searchInput?.focus();
                }} 
              />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
