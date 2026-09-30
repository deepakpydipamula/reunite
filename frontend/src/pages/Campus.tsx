import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { Icon, IconSprite } from "../components/Icons";
import Brand from "../components/Brand";
import ThemeToggle from "../components/ThemeToggle";
import { api, type Item } from "../api";
import { useAuth } from "../auth";
import { campus, pickupPoints } from "../campus/config";
import { createCampusScene, type CampusScene } from "../campus/scene";
import { nice } from "../lib";

type Intent = "/signin" | "/report/lost";

export default function Campus() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const { user } = useAuth();
  const arrivedByDive = (useLocation().state as { dive?: boolean } | null)?.dive === true;
  const intent: Intent = params.get("next") === "/report/lost" ? "/report/lost" : "/signin";
  // Reporting needs an account: signed-out visitors sign in first and are sent back to the report.
  const goto = intent === "/report/lost" ? (user ? "/report/lost" : "/signin?then=%2Freport%2Flost") : user ? "/home" : "/signin?then=%2Fhome";
  const cta = intent === "/report/lost" ? "Continue your report" : user ? "Open your reports" : "Continue to sign in";

  const mapEl = useRef<HTMLDivElement>(null);
  const scene = useRef<CampusScene | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<"3d" | "top">("3d");
  const [failed, setFailed] = useState(false);
  const [veil, setVeil] = useState(true);
  const [mine, setMine] = useState<Item[]>([]);
  const [ready, setReady] = useState(false);

  // Arrive from the white fade the landing page ends on.
  useEffect(() => {
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setVeil(false)));
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    if (!mapEl.current) return;
    const s = createCampusScene({ host: mapEl.current, dive: arrivedByDive, onPick: setSelected, onFail: () => setFailed(true) });
    scene.current = s;
    setReady(!!s);
    return () => {
      setReady(false);
      scene.current = null;
      s?.destroy();
    };
  }, [arrivedByDive]);

  useEffect(() => {
    scene.current?.select(selected);
  }, [selected]);

  // Your own reports as pins: where you lost things and where you found them.
  useEffect(() => {
    if (user) api.items().then(setMine).catch(() => undefined);
  }, [user]);
  useEffect(() => {
    if (!ready) return;
    scene.current?.setMarkers(
      mine.filter((i) => i.lat != null && i.lon != null).map((i) => ({
        id: i.id, kind: i.kind, lat: i.lat as number, lon: i.lon as number,
        label: `${i.kind === "lost" ? "Lost" : "Found"}: ${nice(i.attributes.category?.value) || i.description.slice(0, 24)}`,
      })),
    );
  }, [mine, ready]);

  const choosePitch = (mode: "3d" | "top") => {
    setView(mode);
    scene.current?.pitch(mode);
  };
  const reset = () => {
    setSelected(null);
    setView("3d");
    scene.current?.reset();
  };

  const picked = pickupPoints.find((p) => p.id === selected);

  return (
    <div className="campus">
      <IconSprite />
      <div className={`fade${veil ? " fade--on" : ""}`} aria-hidden="true" />

      <aside className="campus__panel" aria-labelledby="campus-h">
        <div className="campus__top">
          <Link className="brand" to="/welcome" aria-label="reunite, home">
            <Brand />
          </Link>
          <ThemeToggle />
        </div>

        <div>
          <p className="eyebrow"><Icon name="star" size={12} /> Campus map · {campus.short}</p>
          <h1 id="campus-h">Where to <em>collect.</em></h1>
        </div>
        <p className="lede">{campus.name}, {campus.place}. Found items wait at these spots until they are claimed.</p>

        <ul className="campus__list" role="list">
          {pickupPoints.map((p, i) => (
            <li key={p.id}>
              <button className="loc" type="button" aria-pressed={p.id === selected} onClick={() => setSelected(p.id)}>
                <span className="mono loc__n">{String(i + 1).padStart(2, "0")}</span>
                <span className="loc__body">
                  <strong>{p.name}{p.approximate && <span className="loc__tag">Approximate</span>}</strong>
                  <span className="wrap">{p.note}</span>
                  {p.id === selected && <span className="wrap loc__src">{p.source}</span>}
                </span>
                <Icon name="up-right" size={18} />
              </button>
            </li>
          ))}
        </ul>

        {mine.some((i) => i.lat != null) && (
          <p className="campus__legend"><b className="dot dot--lost">L</b> lost by you <b className="dot dot--found">F</b> found by you</p>
        )}
        {picked && <button className="linkbtn" type="button" onClick={reset} style={{ alignSelf: "flex-start" }}>Whole campus</button>}
        <p className="campus__note">
          The college has not named a lost-and-found desk yet, so these are suggested spots. Bring your college ID when you collect an item.
        </p>

        <div className="campus__actions">
          <button className="btn btn--lost" type="button" onClick={() => navigate(goto)}>
            {cta} <Icon name="up-right" size={18} />
          </button>
          <button className="textlink" type="button" onClick={() => navigate("/welcome")}>
            <Icon name="left" /> Back to globe
          </button>
        </div>
      </aside>

      <div className="campus__map" role="region" aria-label={`3D satellite map of ${campus.short}`}>
        <div ref={mapEl} style={{ position: "absolute", inset: 0 }} />
        {!failed && (
          <>
            <div className="mapctl" role="toolbar" aria-label="Map view">
              <div className="mapctl__seg" role="group" aria-label="Angle">
                <button type="button" aria-pressed={view === "3d"} onClick={() => choosePitch("3d")}>3D</button>
                <button type="button" aria-pressed={view === "top"} onClick={() => choosePitch("top")}>Plan</button>
              </div>
              <button type="button" className="mapctl__btn" onClick={() => scene.current?.turn()} aria-label="Turn the view a quarter">Turn</button>
              <button type="button" className="mapctl__btn" onClick={reset}>Reset</button>
            </div>
            <p className="maphint">Drag to orbit · Shift-drag to pan · Scroll to zoom</p>
          </>
        )}
        <div className="campus__fallback" hidden={!failed}>
          <strong>The 3D map isn't available.</strong>
          <span>Your browser could not start WebGL. Turning on hardware acceleration in the browser settings usually fixes it. The pickup spots are listed beside the map.</span>
        </div>
      </div>
    </div>
  );
}
