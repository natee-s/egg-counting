import { useEffect, useRef, useState } from 'react';
import { 
  Egg, 
  Sparkles, 
  Play, 
  Pause, 
  RotateCcw, 
  Zap, 
  ShieldCheck, 
  Activity, 
  Crown, 
  Trophy 
} from 'lucide-react';

const apiURL = (path) => `${import.meta.env.VITE_API_URL || ''}${path}`;
const EMPTY = { total_count: 0, detection_count: 0, tracked_count: 0, fps: 0 };

export default function App() {
  const canvas = useRef(null);
  const [packet, setPacket] = useState(null);
  const [visible, setVisible] = useState(EMPTY);
  const [connection, setConnection] = useState('connecting');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [popups, setPopups] = useState([]);
  const [isBumping, setIsBumping] = useState(false);

  useEffect(() => {
    let socket, timer, disposed = false, attempts = 0;
    let run = null, cursor = 0, pending = null, decoding = false, presenting = null;
    const image = new Image();

    image.onload = () => {
      const next = presenting;
      if (!disposed && next?.run_id === run && canvas.current) {
        const surface = canvas.current;
        if (surface.width !== image.naturalWidth || surface.height !== image.naturalHeight) {
          surface.width = image.naturalWidth;
          surface.height = image.naturalHeight;
        }
        surface.getContext('2d').drawImage(image, 0, 0);
        setVisible(next.stats);
      }
      decoding = false;
    };
    image.onerror = () => { decoding = false; };

    const presentationTimer = setInterval(() => {
      if (disposed || decoding || !pending) return;
      presenting = pending;
      pending = null;
      decoding = true;
      image.src = presenting.frame_data;
    }, 16);

    function connect() {
      if (disposed) return;
      const url = new URL(import.meta.env.VITE_WS_URL || '/ws', window.location.href);
      url.protocol = ['https:', 'wss:'].includes(url.protocol) ? 'wss:' : 'ws:';
      if (run) url.searchParams.set('run_id', run);
      url.searchParams.set('after', cursor);
      socket = new WebSocket(url);

      socket.onopen = () => { 
        if (!disposed) { 
          attempts = 0; 
          setConnection('connected'); 
        } 
      };

      socket.onmessage = ({ data }) => {
        if (disposed) return;
        let next;
        try { next = JSON.parse(data); } catch { return; }
        if (!next.run_id) return;

        if (run !== next.run_id) {
          run = next.run_id;
          cursor = 0;
          pending = null;
          setVisible(EMPTY);
          setPopups([]);
          const surface = canvas.current;
          surface?.getContext('2d').clearRect(0, 0, surface.width, surface.height);
        }

        for (const event of next.events || []) {
          cursor = Math.max(cursor, event.sequence);
        }

        // Spawn RPG Floating "+1" damage/EXP popups at line crossing positions
        if (next.events && next.events.length > 0) {
          const frameW = next.frame_size?.width || 854;
          const frameH = next.frame_size?.height || 480;
          const newPopups = next.events.map((ev, idx) => ({
            id: `${next.run_id}-${ev.sequence || ev.id}-${Date.now()}-${idx}-${Math.random()}`,
            x: (ev.x / frameW) * 100,
            y: (ev.y / frameH) * 100,
            text: '+1',
          }));

          setPopups((prev) => [...prev.slice(-40), ...newPopups]);
          setIsBumping(true);
          setTimeout(() => setIsBumping(false), 350);
        }

        if (!next.stats) return;
        setPacket(next);
        if (next.frame_data) pending = next;
      };

      socket.onerror = () => socket.close();
      socket.onclose = () => {
        if (disposed) return;
        setConnection('disconnected');
        timer = setTimeout(connect, Math.min(10000, 500 * 2 ** attempts++));
      };
    }

    connect();
    return () => {
      disposed = true;
      clearTimeout(timer);
      clearInterval(presentationTimer);
      image.onload = null;
      image.onerror = null;
      socket?.close();
    };
  }, []);

  const state = packet?.state || 'starting';
  const terminal = ['completed', 'error', 'stopped'].includes(state);
  const action = terminal ? 'replay' : state === 'running' ? 'pause' : 'resume';
  const label = terminal ? 'Replay' : state === 'running' ? 'Pause' : 'Play';
  const enabled = connection === 'connected' && packet?.mode === 'real' && state !== 'starting' && !busy;

  const currentLevel = Math.min(99, Math.floor(visible.total_count / 10) + 1);
  const expPercent = ((visible.total_count % 10) / 10) * 100;
  const frameWidth = packet?.frame_size?.width || 854;
  const frameHeight = packet?.frame_size?.height || 480;

  async function control() {
    if (!enabled) return;
    setBusy(true);
    setError('');
    try {
      const response = await fetch(apiURL('/api/control'), {
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ action }),
      });
      if (!response.ok) {
        const body = await response.json();
        throw new Error(body.detail || 'Control failed (' + response.status + ')');
      }
    } catch (err) { 
      setError(err.message); 
    } finally { 
      setBusy(false); 
    }
  }

  const removePopup = (id) => {
    setPopups((prev) => prev.filter((p) => p.id !== id));
  };

  return (
    <main className="ro-viewport">
      {/* Video & Floating Particles Stage */}
      <div className="ro-stage-container">
        <div 
          className="ro-stage-inner" 
          style={{ aspectRatio: `${frameWidth} / ${frameHeight}` }}
        >
          <canvas 
            ref={canvas} 
            width={frameWidth} 
            height={frameHeight} 
            aria-label="Conveyor camera feed" 
          />
          
          {/* Floating RPG +1 Numbers on screen */}
          <div className="ro-popups-layer" aria-hidden="true">
            {popups.map((p) => (
              <div
                key={p.id}
                className="ro-hit-number"
                style={{ left: `${p.x}%`, top: `${p.y}%` }}
                onAnimationEnd={() => removePopup(p.id)}
              >
                <span className="ro-hit-text">{p.text}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Ragnarok MMORPG Style HUD Card */}
      <section className="ro-hud-card" aria-label="Egg count status">
        {/* Corner Golden Rivets */}
        <span className="ro-corner ro-corner-tl" />
        <span className="ro-corner ro-corner-tr" />
        <span className="ro-corner ro-corner-bl" />
        <span className="ro-corner ro-corner-br" />

        {/* Quest / Header Bar */}
        <div className="ro-hud-header">
          <div className="ro-title-badge">
            <div className="ro-egg-avatar">
              <Egg className="ro-egg-svg" />
              <Sparkles className="ro-sparkle-svg" />
            </div>
            <div className="ro-title-info">
              <span className="ro-quest-sub">QUEST REWARD</span>
              <span className="ro-quest-name">EGG HARVEST</span>
            </div>
          </div>
          <div className="ro-level-badge">
            <Crown className="ro-crown-icon" />
            <span>LV.{currentLevel}</span>
          </div>
        </div>

        {/* Main Counter Display */}
        <div className="ro-count-zone">
          <div className="ro-count-label">
            <Trophy className="ro-trophy-icon" />
            <span>TOTAL COLLECTED</span>
          </div>
          <div className={`ro-count-number ${isBumping ? 'ro-bump' : ''}`}>
            <span className="ro-num-val">{visible.total_count.toLocaleString()}</span>
            <span className="ro-unit-text">EGGS</span>
          </div>
        </div>

        {/* EXP Progress Gauge */}
        <div className="ro-exp-container">
          <div className="ro-exp-meta">
            <span>EXP PROGRESS</span>
            <span>{Math.round(expPercent)}%</span>
          </div>
          <div className="ro-exp-track">
            <div 
              className="ro-exp-fill" 
              style={{ width: `${expPercent}%` }} 
            />
          </div>
        </div>

        {/* Legend Badges */}
        <div className="ro-legend-row">
          <div className="ro-gem-badge ro-gem-track">
            <Activity className="ro-badge-icon" />
            <span>Tracking</span>
          </div>
          <div className="ro-gem-badge ro-gem-counted">
            <Sparkles className="ro-badge-icon" />
            <span>Counted</span>
          </div>
        </div>

        {/* Footer & Controls */}
        <div className="ro-hud-footer">
          <div className="ro-system-status">
            <div className={`ro-status-orb ${state}`} />
            <div className="ro-status-texts">
              <span className="ro-status-name">
                {connection !== 'connected' ? connection : packet?.mode === 'demo' ? 'SIMULATION' : state}
              </span>
              <span className="ro-fps-pill">
                <Zap className="ro-zap-icon" />
                {visible.fps.toFixed(1)} FPS
              </span>
            </div>
          </div>

          <button 
            className={`ro-action-btn ${action}`}
            onClick={control} 
            disabled={!enabled} 
            aria-label={label}
          >
            {busy ? (
              <span className="ro-btn-spinner" />
            ) : action === 'replay' ? (
              <>
                <RotateCcw className="ro-btn-icon" />
                <span>Replay</span>
              </>
            ) : action === 'pause' ? (
              <>
                <Pause className="ro-btn-icon" />
                <span>Pause</span>
              </>
            ) : (
              <>
                <Play className="ro-btn-icon" />
                <span>Start</span>
              </>
            )}
          </button>
        </div>

        {/* Error Notification */}
        {(error || packet?.error) && (
          <div className="ro-error-alert" role="alert">
            <span>⚠️ {error || packet?.error}</span>
          </div>
        )}
      </section>
    </main>
  );
}
