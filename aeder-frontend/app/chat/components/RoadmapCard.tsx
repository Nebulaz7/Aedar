"use client";
import React, { useState } from "react";
import {
  Calendar,
  Edit3,
  ChevronDown,
  ChevronUp,
  Clock,
  CheckSquare,
  Square,
  ExternalLink,
  Download,
  Loader2,
} from "lucide-react";

interface Resource {
  type: string;
  title: string;
  link: string;
}

interface RoadmapNode {
  id: string | number;
  title: string;
  description: string;
  duration?: string;
  resources: Resource[];
}

interface RoadmapStage {
  id: string | number;
  title: string;
  duration?: string;
  nodes: RoadmapNode[];
}

interface RoadmapCardProps {
  title: string;
  stages: RoadmapStage[];
  totalDuration?: string;
  onAddToCalendar?: () => void;
  onModifyPlan?: () => void;
  onExport?: () => void;
  isAddingToCalendar?: boolean;
}

const PhaseItem: React.FC<{
  node: RoadmapNode;
  phaseNumber: number;
}> = ({ node, phaseNumber }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);

  return (
    <div
      className={`border-3 border-black transition-all ${
        isCompleted ? "bg-emerald-50" : "bg-white"
      }`}
    >
      <div
        className="flex items-center gap-3 p-4 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <button
          onClick={(e) => {
            e.stopPropagation();
            setIsCompleted(!isCompleted);
          }}
          className="flex-shrink-0"
        >
          {isCompleted ? (
            <CheckSquare className="w-6 h-6 text-emerald-600" />
          ) : (
            <Square className="w-6 h-6 text-black/40 hover:text-black transition-colors" />
          )}
        </button>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-black text-sm text-black/40">
              {phaseNumber}.
            </span>
            <h4
              className={`font-bold text-black ${
                isCompleted ? "line-through text-black/50" : ""
              }`}
            >
              {node.title}
            </h4>
          </div>
        </div>

        {node.duration && (
          <div className="flex items-center gap-1 px-3 py-1 bg-yellow-300 border-2 border-black text-xs font-bold flex-shrink-0">
            <Clock className="w-3 h-3" />
            {node.duration}
          </div>
        )}

        <button className="p-1">
          {isExpanded ? (
            <ChevronUp className="w-5 h-5" />
          ) : (
            <ChevronDown className="w-5 h-5" />
          )}
        </button>
      </div>

      {isExpanded && (
        <div className="px-4 pb-4 pt-0 border-t-2 border-black/10">
          <p className="text-sm text-black/70 mb-4 pl-9">{node.description}</p>

          {node.resources && node.resources.length > 0 && (
            <div className="pl-9">
              <h5 className="text-xs font-bold text-black/50 uppercase tracking-wide mb-2">
                Resources
              </h5>
              <div className="flex flex-wrap gap-2">
                {node.resources.map((resource, idx) => (
                  <a
                    key={idx}
                    href={resource.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 px-3 py-1 bg-white border-2 border-black text-xs font-bold hover:bg-emerald-100 transition-colors"
                  >
                    <span className="text-black/50">{resource.type}:</span>
                    <span>{resource.title}</span>
                    <ExternalLink className="w-3 h-3" />
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const StageSection: React.FC<{
  stage: RoadmapStage;
  stageIndex: number;
}> = ({ stage, stageIndex }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="mb-6">
      {/* Stage Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-3 mb-3"
      >
        <div className="w-10 h-10 bg-emerald-400 border-3 border-black flex items-center justify-center font-black text-lg">
          {stageIndex + 1}
        </div>
        <h3 className="font-black text-lg text-black flex-1 text-left">
          {stage.title}
        </h3>
        {stage.duration && (
          <span className="px-3 py-1 bg-gray-100 border-2 border-black text-sm font-bold">
            {stage.duration}
          </span>
        )}
        {isExpanded ? (
          <ChevronUp className="w-5 h-5" />
        ) : (
          <ChevronDown className="w-5 h-5" />
        )}
      </button>

      {/* Stage Content */}
      {isExpanded && (
        <div className="ml-5 pl-8 border-l-4 border-black/20 space-y-2">
          {stage.nodes.map((node, nodeIndex) => (
            <PhaseItem key={node.id} node={node} phaseNumber={nodeIndex + 1} />
          ))}
        </div>
      )}
    </div>
  );
};

const RoadmapCard: React.FC<RoadmapCardProps> = ({
  title,
  stages,
  totalDuration,
  onAddToCalendar,
  onModifyPlan,
  onExport,
  isAddingToCalendar = false,
}) => {
  const totalMilestones = stages.reduce(
    (acc, stage) => acc + stage.nodes.length,
    0
  );

  return (
    <div className="bg-white border-4 border-black shadow-[8px_8px_0px_0px_rgba(0,0,0,1)]">
      {/* Card Header */}
      <div className="bg-emerald-400 border-b-4 border-black p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="font-black text-xl text-black uppercase">
              Project Roadmap
            </h2>
            <p className="font-bold text-black/70">{title}</p>
          </div>
          <div className="flex items-center gap-2 text-sm font-bold text-black/70">
            {totalDuration && <span>{totalDuration}</span>}
            <span className="text-black/40">•</span>
            <span>{totalMilestones} Milestones</span>
          </div>
        </div>
      </div>

      {/* Card Content */}
      <div className="p-6">
        {stages.map((stage, index) => (
          <StageSection key={stage.id} stage={stage} stageIndex={index} />
        ))}
      </div>

      {/* Card Footer */}
      <div className="border-t-4 border-black p-4 bg-gray-50">
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            onClick={onAddToCalendar}
            disabled={isAddingToCalendar}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-emerald-400 border-3 border-black font-black uppercase shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isAddingToCalendar ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Calendar className="w-5 h-5" />
            )}
            {isAddingToCalendar ? "Adding..." : "Add to Calendar"}
          </button>

          <button
            onClick={onModifyPlan}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-white border-3 border-black font-black uppercase shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] transition-all"
          >
            <Edit3 className="w-5 h-5" />
            Modify Plan
          </button>

          <button
            onClick={onExport}
            className="px-4 py-3 bg-white border-3 border-black font-bold shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[2px] hover:translate-y-[2px] transition-all"
            title="Export"
          >
            <Download className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default RoadmapCard;
