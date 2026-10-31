import React, { useState, useEffect, useRef } from "react";
import {
  Upload,
  Cpu,
  Layers,
  Activity,
  Sparkles,
  Eye,
  RefreshCw,
  TrendingDown,
  Zap,
  CheckCircle2,
  Sliders,
  Image as ImageIcon,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
} from "recharts";

const API_BASE = "http://localhost:8000/api";
const K_PRESETS = [2, 4, 6, 8, 12, 16, 24, 32];

export default function App() {
  // Estados de control de hiperparámetros
  const [selectedFile, setSelectedFile] = useState(null);
  const [k, setK] = useState(8);
  const [initMethod, setInitMethod] = useState("k-means++");
  const [useMinibatch, setUseMinibatch] = useState(true);

  // Estados de respuesta del modelo y UI
  const [data, setData] = useState(null);
  const [elbowData, setElbowData] = useState(null);
  const [activeClusterMask, setActiveClusterMask] = useState(null);
  const [loadingQuantize, setLoadingQuantize] = useState(false);
  const [loadingElbow, setLoadingElbow] = useState(false);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);

  // Cargar demo sintética al abrir la app
  useEffect(() => {
    loadSyntheticDemo(8);
  }, []);

  // Re-ejecutar cuantización automáticamente cuando cambie K, initMethod o archivo
  useEffect(() => {
    if (selectedFile) {
      runQuantization(selectedFile, k, initMethod, useMinibatch);
    }
  }, [k, initMethod, useMinibatch, selectedFile]);

  const loadSyntheticDemo = async (targetK = k) => {
    setLoadingQuantize(true);
    setError(null);
    setActiveClusterMask(null);
    setSelectedFile(null);
    setElbowData(null);
    try {
      const res = await fetch(`${API_BASE}/demo-sample?k=${targetK}`);
      if (!res.ok) throw new Error("No se pudo conectar con el servidor FastAPI");
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingQuantize(false);
    }
  };

  const runQuantization = async (file, targetK, init, minibatch) => {
    setLoadingQuantize(true);
    setError(null);
    setActiveClusterMask(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("k", targetK);
    formData.append("init_method", init);
    formData.append("use_minibatch", minibatch);

    try {
      const res = await fetch(`${API_BASE}/quantize`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Error al cuantizar la imagen");
      }
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingQuantize(false);
    }
  };

  const handleComputeElbow = async () => {
    if (!selectedFile) {
      setError("Sube una imagen desde tu equipo para calcular su Curva del Codo real.");
      return;
    }
    setLoadingElbow(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/elbow`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Error calculando el método del codo");
      const json = await res.json();
      setElbowData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingElbow(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setElbowData(null);
    }
  };

  const handleKChange = (newK) => {
    setK(newK);
    if (!selectedFile) {
      loadSyntheticDemo(newK);
    }
  };

  // Determinar qué imagen mostrar en el lienzo derecho (Cuantizada completa vs. Máscara de cluster)
  const displayedRightImage = activeClusterMask
    ? activeClusterMask.mask_base64
    : data?.quantized_image_base64;

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 p-4 md:p-6 font-sans">
      {/* HEADER */}
      <header className="max-w-7xl mx-auto mb-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-indigo-600/20 border border-indigo-500/40 rounded-xl text-indigo-400">
            <Cpu className="w-7 h-7" />
          </div>
          <div>
            <h1 className="text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              K-Means Vision Lab
              <span className="text-xs font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2.5 py-0.5 rounded-full">
                Aprendizaje No Supervisado
              </span>
            </h1>
            <p className="text-xs md:text-sm text-slate-400">
              Cuantización Vectorial en <span className="text-indigo-400 font-mono">R³ (RGB)</span> y Segmentación Semántica por Centroides
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg text-sm font-semibold transition shadow-lg shadow-indigo-600/25 cursor-pointer"
          >
            <Upload className="w-4 h-4" />
            Subir Imagen (JPG / PNG)
          </button>

          <button
            onClick={() => loadSyntheticDemo(k)}
            className="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-3 py-2 rounded-lg text-sm transition cursor-pointer"
            title="Cargar figura sintética de prueba"
          >
            <Sparkles className="w-4 h-4 text-amber-400" />
            Demo Sintética
          </button>
        </div>
      </header>

      {error && (
        <div className="max-w-7xl mx-auto mb-6 bg-red-950/50 border border-red-500/40 text-red-200 px-4 py-3 rounded-xl text-sm flex items-center justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-xs underline ml-4">
            Cerrar
          </button>
        </div>
      )}

      {/* MAIN GRID */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* PANEL IZQUIERDO: HIPERPARÁMETROS Y MÉTRICAS */}
        <aside className="lg:col-span-4 space-y-5">
          {/* Card: Hiperparámetros */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-indigo-400" />
                Hiperparámetros del Modelo
              </h2>
              {loadingQuantize && (
                <RefreshCw className="w-4 h-4 text-indigo-400 animate-spin" />
              )}
            </div>

            {/* Selector de K */}
            <div className="mb-5">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-300 font-medium">Número de Centroides (K)</span>
                <span className="font-mono font-bold text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-500/30">
                  K = {k}
                </span>
              </div>
              <input
                type="range"
                min="2"
                max="32"
                value={k}
                onChange={(e) => handleKChange(Number(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer"
              />
              <div className="grid grid-cols-4 gap-1.5 mt-3">
                {K_PRESETS.map((preset) => (
                  <button
                    key={preset}
                    onClick={() => handleKChange(preset)}
                    className={`py-1.5 rounded-lg text-xs font-mono font-semibold border transition cursor-pointer ${
                      k === preset
                        ? "bg-indigo-600 text-white border-indigo-400 shadow-md shadow-indigo-600/20"
                        : "bg-slate-800/70 text-slate-400 border-slate-700/60 hover:bg-slate-800 hover:text-slate-200"
                    }`}
                  >
                    K={preset}
                  </button>
                ))}
              </div>
            </div>

            {/* Inicialización: K-Means++ vs Random */}
            <div className="mb-4">
              <label className="block text-xs text-slate-400 uppercase font-semibold mb-2">
                Inicialización de Centroides
              </label>
              <div className="grid grid-cols-2 gap-2">
                {["k-means++", "random"].map((method) => (
                  <button
                    key={method}
                    onClick={() => setInitMethod(method)}
                    className={`py-2 px-3 rounded-lg text-xs font-medium border transition cursor-pointer ${
                      initMethod === method
                        ? "bg-emerald-600/20 border-emerald-500/50 text-emerald-300 font-semibold"
                        : "bg-slate-800/50 border-slate-700 text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {method === "k-means++" ? "K-Means++ (Óptimo)" : "Random (Clásico)"}
                  </button>
                ))}
              </div>
            </div>

            {/* Algoritmo: MiniBatch vs Lloyd estándar */}
            <div>
              <label className="block text-xs text-slate-400 uppercase font-semibold mb-2">
                Motor de Optimización
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setUseMinibatch(true)}
                  className={`py-2 px-3 rounded-lg text-xs font-medium border transition cursor-pointer ${
                    useMinibatch
                      ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300 font-semibold"
                      : "bg-slate-800/50 border-slate-700 text-slate-400"
                  }`}
                >
                  Mini-Batch K-Means
                </button>
                <button
                  onClick={() => setUseMinibatch(false)}
                  className={`py-2 px-3 rounded-lg text-xs font-medium border transition cursor-pointer ${
                    !useMinibatch
                      ? "bg-indigo-600/20 border-indigo-500/50 text-indigo-300 font-semibold"
                      : "bg-slate-800/50 border-slate-700 text-slate-400"
                  }`}
                >
                  Lloyd Exacto (Full)
                </button>
              </div>
            </div>
          </div>

          {/* Card: Telemetría y Métricas Matemáticas */}
          {data && (
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-400" />
                Métricas de Reconstrucción y Costo
              </h2>

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-950/60 border border-slate-800/80 p-3 rounded-xl">
                  <span className="text-[11px] text-slate-400 block">Compresión Teórica</span>
                  <span className="text-xl font-bold text-emerald-400 font-mono">
                    {data.metrics.theoretical_compression_ratio_pct}%
                  </span>
                  <span className="text-[11px] text-slate-500 block mt-0.5">
                    {data.metrics.bits_per_pixel_original} bits → {data.metrics.bits_per_pixel_compressed} bits/px
                  </span>
                </div>

                <div className="bg-slate-950/60 border border-slate-800/80 p-3 rounded-xl">
                  <span className="text-[11px] text-slate-400 block">Colores Únicos</span>
                  <span className="text-xl font-bold text-indigo-400 font-mono">
                    {data.metrics.unique_colors_original.toLocaleString()} → {data.metrics.k}
                  </span>
                  <span className="text-[11px] text-slate-500 block mt-0.5">
                    Reducción del espacio RGB
                  </span>
                </div>

                <div className="bg-slate-950/60 border border-slate-800/80 p-3 rounded-xl">
                  <span className="text-[11px] text-slate-400 block">Error Cuadrático (MSE)</span>
                  <span className="text-lg font-bold text-amber-400 font-mono">
                    {data.metrics.mse}
                  </span>
                  <span className="text-[11px] text-slate-500 block mt-0.5">
                    PSNR: {data.metrics.psnr_db} dB
                  </span>
                </div>

                <div className="bg-slate-950/60 border border-slate-800/80 p-3 rounded-xl">
                  <span className="text-[11px] text-slate-400 block">Tiempo de Inferencia</span>
                  <span className="text-lg font-bold text-sky-400 font-mono flex items-center gap-1">
                    <Zap className="w-4 h-4" />
                    {data.metrics.execution_time_ms} ms
                  </span>
                  <span className="text-[11px] text-slate-500 block mt-0.5">
                    {data.metrics.total_pixels.toLocaleString()} vectores R³
                  </span>
                </div>
              </div>

              <div className="bg-slate-950/80 border border-slate-800 px-3.5 py-2.5 rounded-xl flex justify-between items-center text-xs">
                <span className="text-slate-400">Inercia Total (WCSS):</span>
                <span className="font-mono text-slate-200 font-semibold">
                  {data.metrics.inertia.toLocaleString()}
                </span>
              </div>
            </div>
          )}
        </aside>

        {/* PANEL DERECHO: VISUALIZACIÓN DE IMÁGENES, PALETA Y GRÁFICAS */}
        <section className="lg:col-span-8 space-y-6">
          {/* Comparador Visual Lado a Lado */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-indigo-400" />
                Comparativa en el Espacio de Color
              </h2>

              {activeClusterMask && (
                <button
                  onClick={() => setActiveClusterMask(null)}
                  className="flex items-center gap-1.5 text-xs bg-amber-500/20 border border-amber-500/40 text-amber-300 px-3 py-1 rounded-full hover:bg-amber-500/30 transition cursor-pointer"
                >
                  <Eye className="w-3.5 h-3.5" />
                  Mostrando Máscara del Cluster #{activeClusterMask.cluster_id} ({activeClusterMask.hex}) — Clic para ver todos
                </button>
              )}
            </div>

            {data ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Imagen Original */}
                <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
                  <div className="px-3 py-2 bg-slate-900/80 border-b border-slate-800 flex justify-between items-center text-xs">
                    <span className="font-semibold text-slate-300">Matriz Original (24-bit RGB)</span>
                    <span className="font-mono text-slate-400">
                      {data.metrics.unique_colors_original.toLocaleString()} colores
                    </span>
                  </div>
                  <div className="p-3 flex items-center justify-center min-h-[260px] max-h-[360px]">
                    <img
                      src={data.original_image_base64}
                      alt="Original"
                      className="max-h-[330px] w-auto object-contain rounded-lg"
                    />
                  </div>
                </div>

                {/* Imagen Cuantizada o Máscara de Segmentación */}
                <div className="bg-slate-950 border border-indigo-500/30 rounded-xl overflow-hidden flex flex-col relative">
                  <div className="px-3 py-2 bg-indigo-950/30 border-b border-indigo-500/20 flex justify-between items-center text-xs">
                    <span className="font-semibold text-indigo-300">
                      {activeClusterMask
                        ? `Segmentación Semántica: Cluster #${activeClusterMask.cluster_id}`
                        : `Reconstruida por Centroides (K = ${data.metrics.k})`}
                    </span>
                    <span className="font-mono text-emerald-400">
                      {activeClusterMask
                        ? `${activeClusterMask.percentage}% del área`
                        : `-${data.metrics.theoretical_compression_ratio_pct}% bits`}
                    </span>
                  </div>
                  <div className="p-3 flex items-center justify-center min-h-[260px] max-h-[360px]">
                    <img
                      src={displayedRightImage}
                      alt="Cuantizada"
                      className="max-h-[330px] w-auto object-contain rounded-lg transition-all duration-200"
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center text-slate-500 text-sm">
                Cargando motor de cuantización...
              </div>
            )}

            {/* PALETA INTERACTIVA DE CENTROIDES (SELECTOR DE MÁSCARAS) */}
            {data && (
              <div className="mt-5 pt-4 border-t border-slate-800">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    Centroides Descubiertos (Haz clic en un color para aislar su máscara de segmentación)
                  </span>
                </div>

                {/* Barra Proporcional de Dominancia */}
                <div className="h-6 w-full rounded-lg overflow-hidden flex border border-slate-700 mb-3 shadow-inner">
                  {data.palette.map((item) => (
                    <div
                      key={item.cluster_id}
                      onClick={() =>
                        setActiveClusterMask(
                          activeClusterMask?.cluster_id === item.cluster_id ? null : item
                        )
                      }
                      style={{
                        width: `${Math.max(item.percentage, 1.5)}%`,
                        backgroundColor: item.hex,
                      }}
                      className="h-full cursor-pointer transition-opacity hover:opacity-80"
                      title={`Cluster #${item.cluster_id}: ${item.hex} (${item.percentage}%)`}
                    />
                  ))}
                </div>

                {/* Tarjetas de Centroides */}
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2 max-h-40 overflow-y-auto pr-1">
                  {data.palette.map((item) => {
                    const isSelected = activeClusterMask?.cluster_id === item.cluster_id;
                    return (
                      <button
                        key={item.cluster_id}
                        onClick={() =>
                          setActiveClusterMask(isSelected ? null : item)
                        }
                        className={`flex items-center gap-2 p-2 rounded-xl border text-left transition cursor-pointer ${
                          isSelected
                            ? "bg-indigo-600/25 border-indigo-400 ring-1 ring-indigo-400"
                            : "bg-slate-950/70 border-slate-800 hover:border-slate-600"
                        }`}
                      >
                        <span
                          className="w-6 h-6 rounded-lg shrink-0 border border-white/20 shadow"
                          style={{ backgroundColor: item.hex }}
                        />
                        <div className="overflow-hidden">
                          <p className="text-xs font-mono font-bold text-slate-200 truncate">
                            {item.hex.toUpperCase()}
                          </p>
                          <p className="text-[11px] text-slate-400 font-mono">
                            {item.percentage}%
                          </p>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* FILA INFERIOR: CURVA DEL CODO + DISTRIBUCIÓN VECTORIAL RGB */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Card: Método del Codo (Elbow Curve) */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                      <TrendingDown className="w-4 h-4 text-emerald-400" />
                      Método del Codo (Inercia WCSS)
                    </h3>
                    <p className="text-xs text-slate-400">
                      Evalúa K ∈ [2..32] y detecta el K óptimo geométricamente
                    </p>
                  </div>
                  <button
                    onClick={handleComputeElbow}
                    disabled={loadingElbow}
                    className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold px-3 py-1.5 rounded-lg transition cursor-pointer shrink-0"
                  >
                    {loadingElbow ? "Calculando..." : "Calcular Codo"}
                  </button>
                </div>

                {elbowData ? (
                  <>
                    <div className="mb-3 bg-emerald-950/40 border border-emerald-500/30 rounded-lg px-3 py-2 flex items-center justify-between text-xs">
                      <span className="text-emerald-300 flex items-center gap-1.5">
                        <CheckCircle2 className="w-4 h-4" />
                        Codo detectado (Kneedle): <strong>K = {elbowData.recommended_k}</strong>
                      </span>
                      <button
                        onClick={() => setK(elbowData.recommended_k)}
                        className="underline text-emerald-400 font-semibold cursor-pointer"
                      >
                        Aplicar K={elbowData.recommended_k}
                      </button>
                    </div>

                    <div className="h-52 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={elbowData.elbow_curve}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis
                            dataKey="k"
                            stroke="#94a3b8"
                            tick={{ fontSize: 11 }}
                            label={{ value: "Clusters (K)", position: "insideBottom", offset: -3, fill: "#94a3b8", fontSize: 11 }}
                          />
                          <YAxis
                            stroke="#94a3b8"
                            tick={{ fontSize: 11 }}
                            label={{ value: "Inercia (M)", angle: -90, position: "insideLeft", fill: "#94a3b8", fontSize: 11 }}
                          />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: "#0f172a",
                              borderColor: "#334155",
                              borderRadius: "8px",
                              fontSize: "12px",
                            }}
                          />
                          <ReferenceLine
                            x={elbowData.recommended_k}
                            stroke="#10b981"
                            strokeDasharray="4 4"
                            label={{ value: "Óptimo", fill: "#10b981", fontSize: 11 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="inertia"
                            name="Inercia (Millones)"
                            stroke="#6366f1"
                            strokeWidth={2.5}
                            dot={{ r: 4, fill: "#818cf8" }}
                            activeDot={{ r: 6 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </>
                ) : (
                  <div className="h-52 flex flex-col items-center justify-center text-center p-4 border border-dashed border-slate-800 rounded-xl text-slate-500 text-xs">
                    Sube una imagen y pulsa "Calcular Codo" para entrenar 8 modelos en paralelo y graficar la caída de la inercia.
                  </div>
                )}
              </div>
            </div>

            {/* Card: Proyección del Espacio Vectorial RGB */}
            <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl">
              <div className="mb-3">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                  Espacio de Características (Rojo vs Verde, Z=Azul)
                </h3>
                <p className="text-xs text-slate-400">
                  Muestra de 500 píxeles coloreados según el centroide que los gobierna
                </p>
              </div>

              {data?.rgb_scatter_sample ? (
                <div className="h-56 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: -15 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis
                        type="number"
                        dataKey="r"
                        name="Canal Rojo (R)"
                        domain={[0, 255]}
                        stroke="#94a3b8"
                        tick={{ fontSize: 11 }}
                      />
                      <YAxis
                        type="number"
                        dataKey="g"
                        name="Canal Verde (G)"
                        domain={[0, 255]}
                        stroke="#94a3b8"
                        tick={{ fontSize: 11 }}
                      />
                      <ZAxis type="number" dataKey="b" range={[25, 140]} name="Canal Azul (B)" />
                      <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        contentStyle={{
                          backgroundColor: "#0f172a",
                          borderColor: "#334155",
                          borderRadius: "8px",
                          fontSize: "12px",
                        }}
                      />
                      <Scatter data={data.rgb_scatter_sample} fill="#8884d8">
                        {data.rgb_scatter_sample.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.hex} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-56 flex items-center justify-center text-slate-500 text-xs">
                  Sin muestra de datos vectoriales
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}