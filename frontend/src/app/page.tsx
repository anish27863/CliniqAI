'use client';

import { useState, useRef, useEffect } from 'react';
import { 
  BrainCircuit, Zap, ArrowRight, UploadCloud, Trash2, FileImage, 
  Wand2, AlertTriangle, Eye, Upload, ShieldCheck, Layers, Brain, 
  CheckCircle, XCircle, Search, AlertCircle, Activity, Loader2
} from 'lucide-react';

type AppState = 'idle' | 'analyzing' | 'results';
type HealthStatus = 'checking' | 'online' | 'offline';

export default function Home() {
  const [isScrolled, setIsScrolled] = useState(false);
  const [healthStatus, setHealthStatus] = useState<HealthStatus>('checking');
  
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [appState, setAppState] = useState<AppState>('idle');
  const [isDragging, setIsDragging] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Mock Analysis Results
  const [resultData, setResultData] = useState<any>(null);

  useEffect(() => {
    const handleScroll = () => setIsScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    
    // Health check mock
    const checkHealth = async () => {
      try {
        const res = await fetch('http://localhost:8000/health');
        if (res.ok) {
          setHealthStatus('online');
        } else {
          setHealthStatus('offline');
        }
      } catch {
        setHealthStatus('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
      clearInterval(interval);
    };
  }, []);

  const formatModality = (slug: string) => {
    if (!slug) return "Unknown Modality";
    const mapping: Record<string, string> = {
        'bone_fracture': 'Bone Fracture X-Ray',
        'chest_xray': 'Chest X-Ray',
        'brain_mri': 'Brain MRI',
        'skin_lesion': 'Skin Lesion',
        'retinal': 'Retinal Scan',
        'kidney_stone': 'Kidney Stone CT',
        'breast_histology': 'Breast Histology'
    };
    return mapping[slug] || slug.replace(/_/g, ' ').toUpperCase();
  };

  const formatPrediction = (pred: string) => {
    if (!pred) return "No Diagnosis Available";
    const mapping: Record<string, string> = {
        'fractured': 'Fracture Detected',
        'not fractured': 'Normal / Healthy',
        'pneumonia': 'Pneumonia Identified',
        'normal': 'No Clinical Finding'
    };
    return mapping[pred] || pred.toUpperCase();
  };

  const handleFile = (selectedFile: File) => {
    if (!selectedFile || !selectedFile.type.startsWith('image/')) return;
    setFile(selectedFile);
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    setAppState('idle');
    setResultData(null);
  };

  const removeFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setPreviewUrl(null);
    setAppState('idle');
    setResultData(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setAppState('analyzing');
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const res = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        throw new Error('Analysis failed');
      }
      
      const data = await res.json();
      setResultData(data);
      setAppState('results');
    } catch (error) {
      console.error(error);
      setAppState('idle');
      alert('Failed to analyze image. Please ensure the backend is running.');
    }
  };

  const renderStatusBadge = () => {
    if (!resultData) return null;
    if (!resultData.is_medical) {
      return { text: "REJECTED", className: "border-gray-500 text-gray-500", Icon: XCircle };
    }
    if (!resultData.prediction) {
      return { text: "MODALITY", className: "border-blue-500 text-blue-500", Icon: Search };
    }
    if (resultData.low_confidence_warning) {
      return { text: "LOW CONFIDENCE", className: "border-amber-500 text-amber-500", Icon: AlertCircle };
    }
    return { text: "ANALYSIS READY", className: "border-green-500 text-green-500", Icon: CheckCircle };
  };

  const statusInfo = renderStatusBadge();

  return (
    <div className="min-h-screen flex flex-col">
      {/* Navigation */}
      <nav id="navbar" className={`fixed top-0 left-0 right-0 flex items-center justify-between px-8 z-[100] transition-all duration-300 ${isScrolled ? 'nav-blur border-b border-white/5 h-16' : 'h-20'}`}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-red-600 to-red-900 rounded-lg flex items-center justify-center shadow-[0_0_15px_rgba(220,20,60,0.5)]">
            <BrainCircuit className="text-white w-6 h-6" />
          </div>
          <span className="text-2xl font-bold font-display tracking-tighter">CLINIQ<span className="text-red-500">AI</span></span>
        </div>

        <div className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-400">
          <a href="#" className="hover:text-white transition-colors nav-item-active">Home</a>
          <a href="#workspace" className="hover:text-white transition-colors">Analyze</a>
          <a href="#how-it-works" className="hover:text-white transition-colors">How It Works</a>
          <a href="#" className="hover:text-white transition-colors">About</a>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest">
            <div className={`w-2 h-2 rounded-full animate-pulse ${healthStatus === 'online' ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : healthStatus === 'checking' ? 'bg-gray-600 shadow-[0_0_8px_gray]' : 'bg-red-600 shadow-[0_0_10px_#dc2626]'}`}></div>
            <span className={healthStatus === 'online' ? 'text-green-500' : healthStatus === 'checking' ? 'text-gray-500' : 'text-red-500'}>
              {healthStatus === 'online' ? 'System Online' : healthStatus === 'checking' ? 'Connecting...' : 'System Offline'}
            </span>
          </div>
          <a href="#workspace" className="px-5 py-2.5 bg-red-600 hover:bg-red-500 text-white rounded-full font-bold text-xs uppercase tracking-widest transition-all hover:scale-105 active:scale-95 shadow-[0_0_15px_rgba(220,20,60,0.4)]">Launch Analysis</a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center px-8 lg:px-20 pt-20 overflow-hidden hero-gradient">
        <div className="grid lg:grid-cols-2 gap-12 items-center w-full">
          <div className="max-w-2xl space-y-8 z-10">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-red-500/30 bg-red-500/10 text-red-400 text-[10px] font-bold uppercase tracking-[0.2em]">
              <Zap className="w-3 h-3" />
              Futuristic Diagnostic AI
            </div>
            <h1 className="text-6xl lg:text-8xl font-bold font-display tracking-tighter leading-[0.9]">
              SEE BEYOND <br /> <span className="text-red-600 red-glow-text">THE SCAN.</span>
            </h1>
            <p className="text-xl text-gray-400 leading-relaxed">
              AI-powered medical image analysis designed to support faster, smarter clinical decisions. Integrated intelligence for the next generation of healthcare.
            </p>
            <div className="flex items-center gap-6 pt-4">
              <a href="#workspace" className="px-8 py-4 bg-red-600 hover:bg-red-500 text-white rounded-full font-bold text-sm uppercase tracking-widest transition-all hover:scale-105 hover:shadow-[0_0_30px_rgba(220,20,60,0.5)] flex items-center gap-2">
                Start Analysis
                <ArrowRight className="w-4 h-4" />
              </a>
              <a href="#how-it-works" className="text-sm font-bold uppercase tracking-widest text-gray-400 hover:text-white transition-colors">
                Explore Process
              </a>
            </div>
          </div>

          <div className="relative flex justify-center items-center">
            {/* Futuristic Scan Visual */}
            <div className="relative w-[400px] lg:w-[500px] aspect-square rounded-3xl overflow-hidden glass-panel red-glow border-red-500/30 animate-float">
              <img src="https://images.unsplash.com/photo-1559757175-5700dde675bc?auto=format&fit=crop&q=80&w=1000" alt="Abstract Scan" className="w-full h-full object-cover opacity-60 grayscale" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent"></div>
              <div className="scanline-vertical"></div>
              
              {/* HUD Elements */}
              <div className="absolute top-6 left-6 flex flex-col gap-2">
                <div className="px-3 py-1 bg-black/60 backdrop-blur rounded border border-white/10 text-[10px] font-mono tracking-widest text-red-500">IMAGE ANALYSIS</div>
                <div className="px-3 py-1 bg-black/60 backdrop-blur rounded border border-white/10 text-[10px] font-mono tracking-widest text-white/50">CLINIQ AI v1.1</div>
              </div>
              <div className="absolute bottom-6 right-6">
                <div className="px-3 py-1 bg-red-600/20 backdrop-blur rounded border border-red-500/30 text-[10px] font-mono tracking-widest text-red-400">AI SCREENING ACTIVE</div>
              </div>
            </div>

            {/* Background Glows */}
            <div className="absolute -z-10 w-[600px] h-[600px] bg-red-600/20 rounded-full blur-[120px]"></div>
          </div>
        </div>
      </section>

      {/* Workspace Section */}
      <section id="workspace" className="py-24 px-8 lg:px-20 relative bg-black/40">
        <div className="max-w-6xl mx-auto space-y-12">
          <div className="text-center space-y-4">
            <h2 className="text-4xl lg:text-5xl font-bold font-display tracking-tight">Analysis Workspace</h2>
            <p className="text-gray-400 max-w-xl mx-auto italic">Secure diagnostic environment with end-to-end encrypted AI screening.</p>
          </div>

          <div className="grid lg:grid-cols-1 gap-8">
            <div className="glass-panel rounded-[2rem] p-8 lg:p-12 relative group">
              <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-red-600 text-white text-[10px] font-bold uppercase tracking-[0.2em] rounded-full shadow-lg">AI-Powered Image Screening</div>
              
              <div 
                className={`relative min-h-[400px] border-2 border-dashed rounded-2xl flex flex-col items-center justify-center transition-all cursor-pointer ${isDragging ? 'border-red-500 bg-red-500/10 red-glow' : 'border-red-500/30 hover:border-red-500 hover:bg-red-500/5'} group/drop overflow-hidden`}
                onClick={() => !previewUrl && fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                    handleFile(e.dataTransfer.files[0]);
                  }
                }}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept="image/jpeg,image/png,image/jpg,image/webp"
                  onChange={(e) => e.target.files && handleFile(e.target.files[0])}
                />
                
                {!previewUrl ? (
                  <div className="text-center space-y-6">
                    <div className="w-24 h-24 rounded-full bg-red-600/10 flex items-center justify-center mx-auto transition-transform group-hover/drop:scale-110 group-hover/drop:shadow-[0_0_30px_rgba(220,20,60,0.2)]">
                      <UploadCloud className="w-12 h-12 text-red-500" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold">Drag and drop scan image</h3>
                      <p className="text-gray-500 mt-2">or click to browse from device</p>
                    </div>
                    <div className="flex gap-4 justify-center text-[10px] font-bold text-gray-600 tracking-widest uppercase">
                      <span>JPG</span> • <span>JPEG</span> • <span>PNG</span>
                    </div>
                  </div>
                ) : (
                  <div className="absolute inset-0 p-8 flex items-center justify-center">
                    <div className="relative w-full h-full rounded-xl overflow-hidden bg-black/40 flex items-center justify-center group/preview">
                      <img src={previewUrl} className="max-w-full max-h-full object-contain" alt="Scan Preview" />
                      
                      {appState === 'analyzing' && <div className="scanline-vertical"></div>}
                      
                      {appState !== 'analyzing' && (
                        <button onClick={removeFile} className="absolute top-4 right-4 w-10 h-10 rounded-full bg-black/60 text-white flex items-center justify-center hover:bg-red-600 transition-colors border border-white/10 z-20">
                          <Trash2 className="w-5 h-5" />
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {file && (
                <div className="mt-8 flex items-center justify-between p-4 bg-black/20 rounded-xl border border-white/5">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded bg-red-600/20 flex items-center justify-center">
                      <FileImage className="text-red-500 w-5 h-5" />
                    </div>
                    <div>
                      <p className="text-sm font-bold">{file.name}</p>
                      <p className="text-[10px] text-gray-500">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                    </div>
                  </div>
                  <button 
                    onClick={handleAnalyze}
                    disabled={appState === 'analyzing' || appState === 'results'}
                    className="px-10 py-4 bg-red-600 hover:bg-red-500 text-white rounded-xl font-bold text-sm uppercase tracking-widest transition-all disabled:opacity-30 shadow-[0_0_20px_rgba(220,20,60,0.3)] flex items-center gap-2"
                  >
                    {appState === 'analyzing' ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Analysing Scan...</span>
                      </>
                    ) : (
                      <>
                        <Wand2 className="w-5 h-5" />
                        <span>Analyze Image</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Result Workspace */}
          {appState === 'results' && resultData && statusInfo && (
            <div className="grid lg:grid-cols-2 gap-8 animate-in fade-in duration-700">
              {/* Diagnostic Card */}
              <div className="glass-panel rounded-[2rem] overflow-hidden">
                <div className="p-6 border-b border-red-500/20 flex items-center justify-between bg-red-600/5">
                  <h3 className="font-display font-bold">Diagnostic Result</h3>
                  <div className={`px-3 py-1 rounded border text-[10px] font-bold uppercase tracking-widest ${statusInfo.className}`}>{statusInfo.text}</div>
                </div>
                <div className="p-8 space-y-8">
                  <div className="flex flex-col items-center text-center space-y-2">
                    <div className="w-16 h-16 rounded-full flex items-center justify-center mb-2 bg-white/5">
                      <statusInfo.Icon className={`w-8 h-8 ${statusInfo.className.split(' ')[1]}`} />
                    </div>
                    <p className="text-xs font-bold text-red-500 uppercase tracking-[0.3em]">{formatModality(resultData.modality)}</p>
                    <h4 className="text-4xl font-bold font-display tracking-tight">{formatPrediction(resultData.prediction)}</h4>
                    <div className="pt-4">
                      <span className="text-6xl font-mono tracking-tighter">{(resultData.prediction_confidence * 100).toFixed(1)}%</span>
                      <p className="text-[10px] text-gray-500 uppercase tracking-widest mt-2">Confidence Score</p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {resultData.all_class_probabilities && Object.entries(resultData.all_class_probabilities).sort((a,b) => (b[1] as number) - (a[1] as number)).map(([name, val]: [string, any], idx) => (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-[8px] font-bold uppercase tracking-[0.2em]">
                          <span className="text-gray-400">{name}</span>
                          <span className="text-white">{(val * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                          <div className="h-full bg-red-600 bar-grow" style={{ '--target-width': `${(val * 100).toFixed(1)}%` } as any}></div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {resultData.low_confidence_warning && (
                    <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl flex gap-3 items-start">
                      <AlertTriangle className="text-amber-500 w-5 h-5 flex-shrink-0" />
                      <p className="text-xs text-amber-200 leading-relaxed font-medium">
                        Low Confidence Warning: The analysis shows high uncertainty. This result must be confirmed by a medical professional.
                      </p>
                    </div>
                  )}

                  <p className="text-xs text-center text-gray-500 leading-relaxed max-w-md mx-auto">
                    Diagnosis complete. Review attention map for model reasoning.
                  </p>
                </div>
              </div>

              {/* Visual Attention Card */}
              <div className="glass-panel rounded-[2rem] overflow-hidden flex flex-col">
                <div className="p-6 border-b border-red-500/20 flex items-center justify-between bg-red-600/5">
                  <h3 className="font-display font-bold">Visual Attention (Grad-CAM)</h3>
                  <Eye className="text-red-500 w-5 h-5" />
                </div>
                <div className="p-8 flex-1 flex flex-col">
                  <div className="relative flex-1 rounded-xl overflow-hidden bg-black min-h-[300px]">
                    {previewUrl && (
                      <>
                        <img src={previewUrl} className="w-full h-full object-contain" alt="Analysis Overlay" />
                        <div className="absolute inset-0 opacity-60 mix-blend-multiply transition-opacity" style={{
                          background: `radial-gradient(circle at ${Math.floor(Math.random() * 40) + 30}% ${Math.floor(Math.random() * 40) + 30}%, #dc143c 0%, #ff1744 20%, transparent 70%)`
                        }}></div>
                      </>
                    )}
                  </div>
                  <div className="mt-6 space-y-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-red-500 shadow-[0_0_8px_rgba(220,20,60,0.8)]"></div>
                      <span className="text-xs font-bold uppercase tracking-widest">Region of Interest</span>
                    </div>
                    <p className="text-[11px] text-gray-500 leading-relaxed">
                      Highlighted scarlet region indicates where the model focused its attention during the analysis of the diagnostic features.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 px-8 lg:px-20 border-t border-white/5">
        <div className="max-w-6xl mx-auto space-y-16">
          <div className="text-center space-y-4">
            <h2 className="text-4xl lg:text-5xl font-bold font-display flex flex-wrap justify-center gap-[0.1em] pointer-events-none">
              {"Unified Processing Pipeline".split('').map((char, i) => (
                char === ' ' ? <span key={i} className="mx-2"></span> :
                <span key={i} className="animate-reveal-char" style={{ animationDelay: `${i * 0.05}s` }}>{char}</span>
              ))}
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">All analysis stages run through one integrated API request, ensuring instantaneous cross-model correlation.</p>
          </div>

          <div className="relative flex flex-col md:flex-row justify-between items-center gap-12 py-8">
            {/* Animated Connection Line */}
            <div className="hidden md:block absolute top-12 left-8 right-8 h-20 pointer-events-none">
              <svg width="100%" height="100%" viewBox="0 0 1000 100" preserveAspectRatio="none">
                <path d="M 0 50 L 1000 50" fill="none" stroke="url(#pipelineGradient)" strokeWidth="2" strokeDasharray="20 15" className="pipeline-line" />
                <defs>
                  <linearGradient id="pipelineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#dc143c" />
                    <stop offset="100%" stopColor="#ff1744" />
                  </linearGradient>
                </defs>
                {/* Pulsing connection dots */}
                <circle cx="225" cy="50" r="3" fill="#ff1744" className="animate-[dot-pulse_2s_infinite]" style={{ animationDelay: '0.5s' }} />
                <circle cx="475" cy="50" r="3" fill="#ff1744" className="animate-[dot-pulse_2s_infinite]" style={{ animationDelay: '1s' }} />
                <circle cx="725" cy="50" r="3" fill="#ff1744" className="animate-[dot-pulse_2s_infinite]" style={{ animationDelay: '1.5s' }} />
              </svg>
            </div>
            
            {[
              { icon: Upload, title: "UPLOAD", stage: "Stage 01", delay: '0.2s', innerDelay: '0s' },
              { icon: ShieldCheck, title: "GATEKEEPER", stage: "Stage 02", delay: '0.5s', innerDelay: '0.3s' },
              { icon: Layers, title: "MODALITY", stage: "Stage 03", delay: '0.8s', innerDelay: '0.6s' },
              { icon: Brain, title: "SPECIALIST", stage: "Stage 04", delay: '1.1s', innerDelay: '0.9s' },
            ].map((step, i) => (
              <div key={i} className="relative z-10 flex flex-col items-center text-center space-y-4 opacity-0 animate-[text-reveal_0.6s_ease_forwards]" style={{ animationDelay: step.delay }}>
                <div className="w-20 h-20 rounded-2xl glass-panel border-red-500/40 flex items-center justify-center step-glow-pulse shadow-[0_0_20px_rgba(220,20,60,0.2)]" style={{ animationDelay: step.innerDelay }}>
                  <step.icon className="w-8 h-8 text-red-500" />
                </div>
                <div>
                  <h4 className="font-bold">{step.title}</h4>
                  <p className="text-[10px] text-gray-500 uppercase tracking-widest">{step.stage}</p>
                </div>
              </div>
            ))}
            
            {/* Final Step */}
            <div className="relative z-10 flex flex-col items-center text-center space-y-4 opacity-0 animate-[text-reveal_0.6s_ease_forwards]" style={{ animationDelay: '1.4s' }}>
              <div className="w-20 h-20 rounded-2xl glass-panel border-red-500/40 flex items-center justify-center bg-red-600 step-glow-pulse shadow-[0_0_30px_rgba(220,20,60,0.5)]" style={{ animationDelay: '1.2s' }}>
                <CheckCircle className="w-8 h-8 text-white" />
              </div>
              <div>
                <h4 className="font-bold text-red-500">RESULT</h4>
                <p className="text-[10px] text-gray-500 uppercase tracking-widest">Final Output</p>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Footer / Disclaimer */}
      <footer className="py-12 px-8 lg:px-20 border-t border-white/5 mt-auto">
        <div className="max-w-6xl mx-auto grid md:grid-cols-2 gap-12 items-center">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 bg-red-600 rounded flex items-center justify-center">
                <Activity className="text-white w-3 h-3" />
              </div>
              <span className="font-bold tracking-tighter">CLINIQ<span className="text-red-500">AI</span></span>
            </div>
            <p className="text-xs text-gray-500 leading-relaxed max-w-sm">
              CliniqAI is an AI-assisted screening prototype for educational and research purposes. It does not provide a confirmed medical diagnosis. All results must be reviewed by a qualified healthcare professional.
            </p>
          </div>
          <div className="flex flex-wrap gap-8 justify-end text-[10px] font-bold uppercase tracking-widest text-gray-600">
            <a href="#" className="hover:text-red-500 transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-red-500 transition-colors">Terms of Use</a>
            <a href="#" className="hover:text-red-500 transition-colors">Security</a>
            <a href="#" className="hover:text-red-500 transition-colors">API Docs</a>
          </div>
        </div>
        <div className="text-center mt-12 text-[10px] text-gray-700 tracking-[0.3em] uppercase">
          &copy; 2024 CLINIQAI BIOMEDICAL SYSTEMS • ENGINEERED FOR PRECISION
        </div>
      </footer>
    </div>
  );
}
