'use client';

import { useState, useRef, useEffect } from 'react';
import { Activity, UploadCloud, BrainCircuit, Eye, AlertTriangle, Wand2, CheckCircle2, FileImage } from 'lucide-react';
import Image from 'next/image';

type HealthStatus = 'checking' | 'healthy' | 'error';
type AppState = 'idle' | 'file_selected' | 'analyzing' | 'results';

interface Prediction {
  label: string;
  confidence: number;
}

export default function Home() {
  const [healthStatus, setHealthStatus] = useState<HealthStatus>('checking');
  const [appState, setAppState] = useState<AppState>('idle');
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  
  // Mock results state
  const [modality, setModality] = useState<string>('');
  const [prediction, setPrediction] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(0);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [lowConfidence, setLowConfidence] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Mock health check
    const checkHealth = async () => {
      try {
        // In a real app, this would be: await fetch('/health')
        await new Promise(resolve => setTimeout(resolve, 1000));
        setHealthStatus('healthy');
      } catch {
        setHealthStatus('error');
      }
    };
    checkHealth();
  }, []);

  const handleFile = (selectedFile: File) => {
    if (selectedFile) {
      setFile(selectedFile);
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
      setAppState('file_selected');
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleAnalyze = async () => {
    setAppState('analyzing');
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2500));
    
    // Mock API response
    const mockModality = 'bone_fracture';
    const mockPrediction = 'fractured';
    const mockConfidence = 87.5;
    
    setModality(formatModality(mockModality));
    setPrediction(formatPrediction(mockPrediction));
    setConfidence(mockConfidence);
    setLowConfidence(mockConfidence < 70);
    setPredictions([
      { label: 'Fractured', confidence: mockConfidence },
      { label: 'Normal', confidence: 100 - mockConfidence }
    ]);
    
    setAppState('results');
  };

  const formatModality = (slug: string) => {
    return slug.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ') + ' X-Ray';
  };

  const formatPrediction = (pred: string) => {
    return pred === 'fractured' ? 'Fracture Detected' : 'Normal';
  };

  return (
    <div className="flex flex-col min-h-screen">
      <header className="h-16 px-6 bg-white border-b border-[#E2E8F0] flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-[#0F1B2D] rounded flex items-center justify-center">
            <Activity className="text-white w-5 h-5" />
          </div>
          <h1 className="text-xl font-bold font-display tracking-tight text-[#0F1B2D]">CliniqAI</h1>
          <span className="px-2 py-0.5 bg-[#E2E8F0] text-[#4A5A70] text-[10px] font-bold rounded uppercase tracking-widest ml-1">v1.1</span>
        </div>
        <div className="flex items-center gap-2 text-sm font-medium text-[#4A5A70]">
          <div className={`w-2 h-2 rounded-full ${healthStatus === 'healthy' ? 'bg-[#059669]' : healthStatus === 'checking' ? 'bg-gray-400' : 'bg-[#D64545]'}`}></div>
          <span>{healthStatus === 'healthy' ? 'System Online' : healthStatus === 'checking' ? 'Checking system...' : 'System Error'}</span>
        </div>
      </header>

      <main className="flex-1 p-6 max-w-7xl mx-auto w-full">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Upload Panel */}
          <section className="bg-white border border-[#E2E8F0] rounded-2xl flex flex-col shadow-sm">
            <div className="p-5 border-b border-[#E2E8F0] flex items-center justify-between">
              <h2 className="font-display font-semibold">1. Upload Scan</h2>
              <UploadCloud className="text-[#4A5A70] w-5 h-5" />
            </div>
            <div className="p-6 flex-1 flex flex-col">
              <div 
                className={`relative flex-1 min-h-[300px] border-2 border-dashed border-[#E2E8F0] rounded-xl flex flex-col items-center justify-center transition-all cursor-pointer hover:border-[#0891A8] hover:bg-[#E0F4F7]/10 group overflow-hidden ${appState === 'analyzing' ? 'opacity-50 pointer-events-none' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDrop={onDrop}
                onDragOver={onDragOver}
              >
                <input 
                  type="file" 
                  ref={fileInputRef}
                  className="hidden" 
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(e) => e.target.files && handleFile(e.target.files[0])}
                />
                
                {previewUrl ? (
                  <div className="absolute inset-0 w-full h-full p-2">
                    <img src={previewUrl} alt="Preview" className="w-full h-full object-contain rounded-lg" />
                    {appState === 'analyzing' && (
                      <div className="absolute inset-0 bg-[#0891A8]/10 scanline"></div>
                    )}
                  </div>
                ) : (
                  <div className="text-center px-4">
                    <UploadCloud className="w-10 h-10 text-[#4A5A70] mx-auto mb-3 group-hover:text-[#0891A8] transition-colors" />
                    <p className="font-medium text-[#0F1B2D]">Click to upload or drag & drop</p>
                    <p className="text-xs text-[#4A5A70] mt-1">JPEG, PNG, WEBP up to 10MB</p>
                  </div>
                )}
              </div>
              
              {file && (
                <div className="mt-4">
                  <div className="flex items-center gap-3 p-3 bg-[#FAFBFC] rounded-lg border border-[#E2E8F0]">
                    <FileImage className="text-[#0891A8] w-5 h-5 flex-shrink-0" />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-[#0F1B2D] truncate">{file.name}</p>
                      <p className="text-xs text-[#4A5A70]">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                  </div>
                </div>
              )}
              
              <button 
                onClick={handleAnalyze}
                disabled={!file || appState === 'analyzing' || appState === 'results'}
                className="mt-6 w-full py-4 rounded-xl bg-[#0891A8] text-white font-bold text-sm tracking-wide disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:bg-[#077E92] shadow-lg shadow-[#0891A8]/20 flex items-center justify-center gap-2"
              >
                {appState === 'analyzing' ? (
                  <div className="pulse-loading flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    <span>Analyzing Scan...</span>
                  </div>
                ) : (
                  <>
                    <Wand2 className="w-5 h-5" />
                    <span>Analyze Scan</span>
                  </>
                )}
              </button>
            </div>
          </section>

          {/* Diagnosis Panel */}
          <section className="bg-white border border-[#E2E8F0] rounded-2xl flex flex-col overflow-hidden shadow-sm">
            <div className="p-5 border-b border-[#E2E8F0] flex items-center justify-between">
              <h2 className="font-display font-semibold">2. Diagnostic Intelligence</h2>
              <BrainCircuit className="text-[#4A5A70] w-5 h-5" />
            </div>
            <div className="p-8 flex-1 flex flex-col justify-center">
              {appState === 'results' ? (
                <div className="h-full flex flex-col animate-in fade-in zoom-in duration-500">
                  {lowConfidence && (
                    <div className="-mt-8 -mx-8 mb-8 p-3 bg-[#FEF3C7] text-[#D97706] text-xs font-semibold flex items-center justify-center gap-2 border-b border-[#D97706]/20">
                      <AlertTriangle className="w-4 h-4" />
                      <span>⚠ Low Confidence — Confirm with a medical professional</span>
                    </div>
                  )}
                  <div className="space-y-6">
                    <div className="text-center">
                      <div className="w-12 h-12 rounded-full flex items-center justify-center mx-auto mb-4 bg-[#E0F4F7] text-[#0891A8]">
                        <CheckCircle2 className="w-6 h-6" />
                      </div>
                      <h3 className="text-2xl font-bold font-display tracking-tight text-[#0F1B2D]">{prediction}</h3>
                      <p className="text-sm text-[#4A5A70] mt-1 uppercase tracking-widest font-medium">{modality}</p>
                    </div>
                    <div className="text-center py-6 bg-[#FAFBFC] rounded-2xl border border-[#E2E8F0]">
                      <p className="text-[10px] text-[#4A5A70] uppercase font-bold tracking-[0.2em] mb-1">Confidence Score</p>
                      <span className="text-5xl font-mono text-[#0F1B2D]">{confidence.toFixed(1)}%</span>
                    </div>
                    <div className="space-y-3">
                      {predictions.map((pred, i) => (
                        <div key={i} className="space-y-1.5">
                          <div className="flex justify-between text-xs font-medium">
                            <span className="text-[#0F1B2D]">{pred.label}</span>
                            <span className="text-[#4A5A70] font-mono">{pred.confidence.toFixed(1)}%</span>
                          </div>
                          <div className="h-2 w-full bg-[#E2E8F0] rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-[#0891A8] rounded-full bar-grow"
                              style={{ '--target-width': `${pred.confidence}%` } as React.CSSProperties}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-[#4A5A70] opacity-50">
                  <BrainCircuit className="w-12 h-12 mx-auto mb-4 opacity-20" />
                  <p className="text-sm font-medium">Awaiting scan for analysis...</p>
                </div>
              )}
            </div>
          </section>

          {/* Heatmap Panel */}
          <section className="bg-white border border-[#E2E8F0] rounded-2xl flex flex-col overflow-hidden shadow-sm">
            <div className="p-5 border-b border-[#E2E8F0] flex items-center justify-between">
              <h2 className="font-display font-semibold">3. Visual Attention</h2>
              <Eye className="text-[#4A5A70] w-5 h-5" />
            </div>
            <div className="p-6 flex-1 flex flex-col">
              {appState === 'results' && previewUrl ? (
                <div className="flex-1 flex flex-col animate-in fade-in zoom-in duration-700">
                  <div className="flex justify-between items-center mb-4">
                    <span className="text-xs font-bold text-[#0F1B2D] uppercase tracking-wider">Grad-CAM Attention Map</span>
                  </div>
                  <div className="relative flex-1 bg-black rounded-lg overflow-hidden min-h-[250px] flex items-center justify-center">
                    <img src={previewUrl} className="w-full h-full object-contain opacity-60" alt="Heatmap base" />
                    {/* Simulated Heatmap Overlay */}
                    <div className="absolute inset-0 mix-blend-screen pointer-events-none" 
                      style={{
                        background: 'radial-gradient(circle at 50% 50%, rgba(255, 0, 0, 0.6) 0%, rgba(255, 165, 0, 0.3) 30%, transparent 60%)'
                      }}
                    ></div>
                  </div>
                  <div className="mt-6 space-y-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-red-500"></div>
                      <span className="text-sm font-bold text-[#0F1B2D]">Region of Interest</span>
                    </div>
                    <div className="p-3 bg-[#FAFBFC] rounded-lg border border-[#E2E8F0]">
                      <p className="text-[11px] text-[#4A5A70] leading-relaxed">
                        Highlighted region indicates model attention. 
                        {lowConfidence && (
                          <span className="font-semibold text-[#D97706] ml-1">
                            Attention map may be less reliable at low confidence.
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-[#4A5A70] opacity-50">
                  <Eye className="w-12 h-12 mx-auto mb-4 opacity-20" />
                  <p className="text-sm font-medium">Attention map will appear here</p>
                </div>
              )}
            </div>
          </section>

        </div>
      </main>
    </div>
  );
}
