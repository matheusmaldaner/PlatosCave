import React, { useState } from "react";
import { computePaperScore } from "../lib/aggregate";

interface ScoreSliderProps {
  label?: string;
}

const ScoreSlider: React.FC<ScoreSliderProps> = ({ label = "Depth Decay (α)" }) => {
  const [alpha, setAlpha] = useState(0.15);
  const [score, setScore] = useState(0);

  // temp static test data
  const nodes = [
    {
      credibility: 0.9,
      relevance: 0.85,
      evidence_strength: 0.8,
      method_rigor: 0.85,
      reproducibility: 0.9,
      citation_support: 0.8,
      role: "result",
      level: 2,
    },
    {
      credibility: 0.95,
      relevance: 0.9,
      evidence_strength: 0.9,
      method_rigor: 0.9,
      reproducibility: 0.9,
      citation_support: 0.95,
      role: "hypothesis",
      level: 0,
    },
  ];

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newAlpha = parseFloat(e.target.value);
    setAlpha(newAlpha);
    const newScore = computePaperScore(nodes, newAlpha);
    setScore(newScore);
  };

  return (
    <div className="mt-12 bg-gray-50 border border-gray-200 rounded-xl shadow-sm p-6 w-full max-w-md text-center">
      <h2 className="text-lg font-semibold text-gray-800 mb-2">{label}</h2>

      <input
        type="range"
        min="0"
        max="0.5"
        step="0.01"
        value={alpha}
        onChange={handleSliderChange}
        className="w-full accent-emerald-600 cursor-pointer h-2 rounded-lg"
      />

      <p className="text-gray-600 mt-3">
        Current α: <span className="font-medium">{alpha.toFixed(2)}</span>
      </p>
    </div>
  );
};

export default ScoreSlider;