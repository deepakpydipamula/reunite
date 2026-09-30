import "cesium/Build/Cesium/Widgets/widgets.css";
import {
  BoundingSphere,
  Cartesian2,
  Cartesian3,
  Color,
  Credit,
  DirectionalLight,
  Entity,
  HeadingPitchRange,
  HeightReference,
  HorizontalOrigin,
  Cartographic,
  Cesium3DTileStyle,
  Ellipsoid,
  Ion,
  LabelStyle,
  Math as CesiumMath,
  Matrix4,
  NearFarScalar,
  DistanceDisplayCondition,
  Terrain,
  Transforms,
  VerticalOrigin,
  Viewer,
  createGooglePhotorealistic3DTileset,
  createOsmBuildingsAsync,
  sampleTerrainMostDetailed,
} from "cesium";
import buildings from "./buildings.json";
import { campus, landmarks, pickupPoints } from "./config";
import { DIVE_HEIGHT } from "../globe/config";
import { addEarthImagery } from "../globe/imagery";

type Building = { id: number; name?: string; h: number; r: number[] };

/** A pin for an item: where it was lost, where it was found, where to collect it, or the spot being chosen. */
export type Marker = { id: string; kind: "lost" | "found" | "collect" | "pick"; lat: number; lon: number; label?: string };

export type CampusScene = {
  select(id: string | null): void;
  /** Replace every item marker. */
  setMarkers(markers: Marker[]): void;
  /** Fly to a point. */
  focus(lat: number, lon: number, range?: number): void;
  /** Tilted 3D view, or straight down like a plan. */
  pitch(mode: "3d" | "top"): void;
  /** Quarter turn around the campus. */
  turn(): void;
  reset(): void;
  destroy(): void;
};

type Options = {
  host: HTMLElement;
  /** Arrived by the globe dive: start low and close, instead of from orbit. */
  dive?: boolean;
  /** Show the pickup pins (the full campus page). Off in the small place pickers. */
  pickups?: boolean;
  /** Camera distance in metres for the whole-campus view. */
  range?: number;
  onPick?(id: string): void;
  /** Called with the ground position when the map is clicked (used to drop a pin). */
  onGround?(lat: number, lon: number): void;
  onFail(): void;
};

const PITCH_3D = CesiumMath.toRadians(-38);
const PITCH_TOP = -CesiumMath.PI_OVER_TWO;
const PITCH_MIN = CesiumMath.toRadians(-20); // never lower than this: the horizon shows nothing useful
const HEADING = CesiumMath.toRadians(-24);
const RANGE = 780; // metres from the campus centre for the whole-campus view
const ROOF = "#ECE9E3";
const ORANGE = "#FF6A13";
const NAME_RANGE = 380; // pin names appear when the camera is this close (metres)

// Cesium's free OSM Buildings are one flat grey everywhere in the world; most buildings carry no colour tag of
// their own. A little per-building variation, in the same warm neutral family as ROOF, reads as a real campus
// instead of a single uniform mass, at no cost.
const BUILDING_STYLE = new Cesium3DTileStyle({
  color: {
    conditions: [
      ["${elementId} % 5 === 0", "color('#EDE9E1')"],
      ["${elementId} % 5 === 1", "color('#E3DED4')"],
      ["${elementId} % 5 === 2", "color('#E9E4DA')"],
      ["${elementId} % 5 === 3", "color('#DED9CE')"],
      ["true", "color('#E6E1D7')"],
    ],
  },
});

const pin = (n: number, on: boolean) => {
  const fill = on ? "#0B0B0B" : ORANGE;
  const text = on ? "#FFFFFF" : "#0B0B0B";
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="40" height="48" viewBox="0 0 40 48">` +
    `<path d="M20 46C20 46 4 30 4 19a16 16 0 0 1 32 0c0 11-16 27-16 27z" fill="${fill}" stroke="#FFFFFF" stroke-width="3"/>` +
    `<text x="20" y="25" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14" font-weight="700" fill="${text}">${n}</text></svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
};

/** True when this browser can give us a WebGL context. Cesium cannot draw anything without one. */
function webglAvailable(): boolean {
  try {
    const probe = document.createElement("canvas");
    const gl = probe.getContext("webgl2") || probe.getContext("webgl") || probe.getContext("experimental-webgl");
    return !!gl;
  } catch {
    return false;
  }
}

export function createCampusScene({ host, dive = false, pickups = true, range: viewRange = RANGE, onPick, onGround, onFail }: Options): CampusScene | null {
  const ion = import.meta.env.VITE_CESIUM_ION_TOKEN as string | undefined;
  const googleKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string | undefined;

  // No WebGL (hardware acceleration off, blocked GPU, some remote desktops): skip Cesium entirely and let the page
  // show its own fallback, instead of Cesium's blocking "Error constructing CesiumWidget" dialog.
  if (!webglAvailable()) {
    console.warn("WebGL is not available in this browser, so the campus map is disabled");
    onFail();
    return null;
  }

  let viewer: Viewer;
  try {
    viewer = new Viewer(host, {
      // We show our own fallback; Cesium's built-in error dialog would cover the page and block the buttons.
      showRenderLoopErrors: false,
      baseLayer: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      fullscreenButton: false,
      infoBox: false,
      selectionIndicator: false,
      requestRenderMode: true,
    });
  } catch (err) {
    console.error("Campus map could not start", err);
    onFail();
    return null;
  }
  const { scene, camera } = viewer;
  if (import.meta.env.DEV) (window as unknown as { __campus: Viewer }).__campus = viewer;
  addEarthImagery(viewer);
  viewer.useBrowserRecommendedResolution = false;
  viewer.resolutionScale = Math.min(window.devicePixelRatio || 1, 2) / (window.devicePixelRatio || 1);
  scene.globe.enableLighting = false;
  scene.renderError.addEventListener(onFail);
  viewer.creditDisplay.addStaticCredit(new Credit('<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">© OpenStreetMap contributors</a>', true));

  const useIon = !!ion;
  const useGoogle = !!googleKey;
  const clamp = useGoogle ? HeightReference.CLAMP_TO_3D_TILE : useIon ? HeightReference.CLAMP_TO_GROUND : HeightReference.NONE;

  // Real elevation, once we know it (see settleGround). Hyderabad's plateau sits around 540m above the ellipsoid,
  // so that is the starting guess with real terrain or tiles, rather than assuming sea level, until the exact
  // figure comes back; against the plain globe (no terrain) the ground truly is at 0.
  let ground = useIon || useGoogle ? 540 : 0;
  const centre = Cartesian3.fromDegrees(campus.lon, campus.lat, ground);
  const sphere = new BoundingSphere(centre, campus.radius);

  // Fixed sun from the south-west, so the blocks keep their shading whatever the time of day.
  const enu = Transforms.eastNorthUpToFixedFrame(centre);
  scene.light = new DirectionalLight({ direction: Cartesian3.normalize(Matrix4.multiplyByPointAsVector(enu, new Cartesian3(0.55, 0.45, -0.7), new Cartesian3()), new Cartesian3()) });

  // ---- 3D interaction. The camera orbits a target point: drag to turn and tilt, shift-drag or two fingers to pan,
  // wheel or pinch to zoom. Cesium's own controllers are off, so the camera cannot dive into the ground or fly away.
  const ctrl = scene.screenSpaceCameraController;
  ctrl.enableRotate = false;
  ctrl.enableTilt = false;
  ctrl.enableTranslate = false;
  ctrl.enableLook = false;
  ctrl.enableZoom = false;

  /** Once we learn the real ground height at the campus, move the orbit target and framing sphere onto it. */
  const settleGround = (h: number | undefined) => {
    if (viewer.isDestroyed() || h === undefined || !Number.isFinite(h)) return;
    ground = h;
    Cartesian3.fromDegrees(campus.lon, campus.lat, ground, undefined, centre);
    Cartesian3.clone(centre, sphere.center);
    apply();
  };

  // ---- buildings, extruded from OpenStreetMap footprints. This is the fallback: real 3D tiles (below) replace it
  // when they load, and it is also what shows up if a real tile source is configured but fails to load.
  const roofs = new Map<number, Entity>();
  let builtFallbackBoxes = false;
  const buildFlatBoxes = () => {
    if (builtFallbackBoxes || viewer.isDestroyed()) return;
    builtFallbackBoxes = true;
    for (const b of buildings as Building[]) {
      roofs.set(
        b.id,
        viewer.entities.add({
          polygon: {
            hierarchy: Cartesian3.fromDegreesArray(b.r),
            // Absolute ellipsoid height, not clamped: with real terrain active this must start from the real
            // ground, or the boxes end up buried under it.
            height: ground,
            extrudedHeight: ground + b.h,
            // Fully opaque: anything below 1 puts the polygon in Cesium's translucent pass, which does not depth-sort
            // separate faces correctly and makes the satellite ground show straight through the walls ("hollow" boxes).
            material: Color.fromCssColorString(ROOF),
            outline: true,
            outlineColor: Color.fromCssColorString("#0B0B0B").withAlpha(0.35),
          },
        }),
      );
    }
    scene.requestRender();
  };

  // With a Cesium ion token the map gets real terrain and Cesium OSM Buildings (real building meshes, worldwide,
  // free with any ion account). Without one, or if either fails to load, the flat footprints above stand in.
  if (useIon) {
    Ion.defaultAccessToken = ion!;
    const terrain = Terrain.fromWorldTerrain();
    terrain.readyEvent.addEventListener((provider) => {
      sampleTerrainMostDetailed(provider, [Cartographic.fromDegrees(campus.lon, campus.lat)])
        .then(([c]) => settleGround(c?.height))
        .catch(() => undefined);
    });
    scene.setTerrain(terrain);
    createOsmBuildingsAsync()
      .then((tiles) => {
        if (viewer.isDestroyed()) return;
        tiles.style = BUILDING_STYLE;
        scene.primitives.add(tiles);
      })
      .catch((err) => {
        console.warn("Cesium OSM Buildings unavailable, falling back to flat footprints", err);
        buildFlatBoxes();
      });
  } else if (!useGoogle) {
    buildFlatBoxes();
  }
  const heightOf = new Map((buildings as Building[]).map((b) => [b.id, b.h]));

  // ---- labels for context
  for (const l of landmarks) {
    viewer.entities.add({
      position: Cartesian3.fromDegrees(l.lon, l.lat),
      label: {
        text: l.name,
        font: "500 12px 'IBM Plex Sans', Helvetica, Arial, sans-serif",
        style: LabelStyle.FILL_AND_OUTLINE,
        fillColor: Color.WHITE,
        outlineColor: Color.fromCssColorString("#0B0B0B").withAlpha(0.85),
        outlineWidth: 4,
        horizontalOrigin: HorizontalOrigin.CENTER,
        verticalOrigin: VerticalOrigin.BOTTOM,
        pixelOffset: new Cartesian2(0, -4),
        heightReference: clamp,
        distanceDisplayCondition: new DistanceDisplayCondition(0, 750),
        translucencyByDistance: new NearFarScalar(450, 1, 750, 0.15),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  }

  // ---- pickup pins, lifted above the roof of the building they belong to
  const pins = (pickups ? pickupPoints : []).map((p, i) => {
    const lift = p.buildingId ? (heightOf.get(p.buildingId) ?? 0) + 3 : 4;
    return viewer.entities.add({
      id: p.id,
      position: Cartesian3.fromDegrees(p.lon, p.lat, useIon || useGoogle ? 0 : lift),
      billboard: {
        image: pin(i + 1, false),
        width: 34,
        height: 41,
        verticalOrigin: VerticalOrigin.BOTTOM,
        heightReference: clamp,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: p.name,
        font: "600 13px 'IBM Plex Sans', Helvetica, Arial, sans-serif",
        style: LabelStyle.FILL,
        fillColor: Color.fromCssColorString("#0B0B0B"),
        showBackground: true,
        backgroundColor: Color.WHITE.withAlpha(0.94),
        backgroundPadding: new Cartesian2(8, 5),
        horizontalOrigin: HorizontalOrigin.LEFT,
        verticalOrigin: VerticalOrigin.BOTTOM,
        pixelOffset: new Cartesian2(20, -14),
        heightReference: clamp,
        // Names crowd the overview, so they show up close, or when the spot is chosen (see paint).
        distanceDisplayCondition: new DistanceDisplayCondition(0, NAME_RANGE),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
  });

  // ---- camera moves
  const orbit = { target: centre, heading: HEADING, pitch: PITCH_3D, range: viewRange };
  const apply = () => {
    camera.lookAt(orbit.target, new HeadingPitchRange(orbit.heading, orbit.pitch, orbit.range));
    camera.lookAtTransform(Matrix4.IDENTITY);
    scene.requestRender();
  };
  const frame = (target: BoundingSphere, heading: number, pitch: number, range: number, duration = 1.4) => {
    Object.assign(orbit, { target: target.center, heading, pitch, range });
    camera.flyToBoundingSphere(target, { offset: new HeadingPitchRange(heading, pitch, range), duration });
  };
  const nowPitch = () => (camera.pitch < CesiumMath.toRadians(-75) ? PITCH_TOP : PITCH_3D);

  // Arrive from above: from the dive height above the campus, or from orbit when opened directly.
  camera.setView({
    destination: Cartesian3.fromDegrees(campus.lon, campus.lat, dive ? DIVE_HEIGHT : 9_000_000),
    orientation: { heading: 0, pitch: PITCH_TOP, roll: 0 },
  });
  frame(sphere, HEADING, PITCH_3D, viewRange, dive ? 2.4 : 3);

  // ---- pointer input
  const canvas = scene.canvas;
  canvas.style.touchAction = "none";
  const pointers = new Map<number, { x: number; y: number }>();
  let moved = 0;
  let pinch = 0;
  const syncFromCamera = () => {
    camera.cancelFlight();
    orbit.heading = camera.heading;
    orbit.pitch = camera.pitch;
    orbit.range = Cartesian3.distance(camera.positionWC, orbit.target);
  };
  const pan = (dx: number, dy: number) => {
    const mpp = (2 * orbit.range * Math.tan(CesiumMath.toRadians(30))) / canvas.clientHeight;
    const east = -(Math.cos(orbit.heading) * dx) * mpp + Math.sin(orbit.heading) * dy * mpp / Math.max(0.3, Math.sin(-orbit.pitch));
    const north = Math.sin(orbit.heading) * dx * mpp + Math.cos(orbit.heading) * dy * mpp / Math.max(0.3, Math.sin(-orbit.pitch));
    const m = Transforms.eastNorthUpToFixedFrame(orbit.target);
    const next = Cartesian3.add(orbit.target, Matrix4.multiplyByPointAsVector(m, new Cartesian3(east, north, 0), new Cartesian3()), new Cartesian3());
    // Stay near the campus, and on the ground.
    const onGround = Ellipsoid.WGS84.scaleToGeodeticSurface(next, new Cartesian3());
    if (onGround && Cartesian3.distance(onGround, centre) < 900) orbit.target = onGround;
  };
  const clampOrbit = () => {
    orbit.pitch = CesiumMath.clamp(orbit.pitch, -CesiumMath.PI_OVER_TWO, PITCH_MIN);
    orbit.range = CesiumMath.clamp(orbit.range, 60, 2500);
  };
  const rect = (_e: PointerEvent) => canvas.getBoundingClientRect();
  const onDown = (e: PointerEvent) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1) moved = 0;
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      pinch = Math.hypot(a.x - b.x, a.y - b.y);
    }
    syncFromCamera();
  };
  const onMove = (e: PointerEvent) => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x;
    const dy = e.clientY - prev.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    moved += Math.abs(dx) + Math.abs(dy);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch > 0 && d > 0) orbit.range *= pinch / d;
      pinch = d;
      pan(dx / 2, dy / 2);
    } else if (e.shiftKey || e.button === 2 || (e.buttons & 2) === 2) {
      pan(dx, dy);
    } else {
      orbit.heading -= dx * 0.006;
      orbit.pitch += dy * 0.005;
    }
    clampOrbit();
    apply();
  };
  const onUp = (e: PointerEvent) => {
    pointers.delete(e.pointerId);
    pinch = 0;
    if (moved < 5 && pointers.size === 0) {
      const r = rect(e);
      const hit = scene.pick(new Cartesian2(e.clientX - r.left, e.clientY - r.top));
      const id = hit?.id instanceof Entity ? String(hit.id.id) : null;
      if (id && pickups && pickupPoints.some((p) => p.id === id)) onPick?.(id);
      else if (onGround) {
        const at = new Cartesian2(e.clientX - r.left, e.clientY - r.top);
        // A roof or 3D tile if there is one under the pointer, otherwise the flat ground.
        const point = (scene.pickPositionSupported && scene.pickPosition(at)) || camera.pickEllipsoid(at, Ellipsoid.WGS84);
        if (point) {
          const c = Cartographic.fromCartesian(point);
          onGround(CesiumMath.toDegrees(c.latitude), CesiumMath.toDegrees(c.longitude));
        }
      }
    }
  };
  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    syncFromCamera();
    orbit.range *= Math.exp(e.deltaY * 0.0012);
    clampOrbit();
    apply();
  };
  const noMenu = (e: Event) => e.preventDefault();
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("contextmenu", noMenu);

  let selectedId: string | null = null;
  const paint = () => {
    pickupPoints.forEach((p, i) => {
      if (!pins[i]) return;
      const on = p.id === selectedId;
      pins[i].billboard!.image = pin(i + 1, on) as never;
      pins[i].label!.distanceDisplayCondition = new DistanceDisplayCondition(0, on ? Number.POSITIVE_INFINITY : NAME_RANGE) as never;
      if (p.buildingId) {
        const roof = roofs.get(p.buildingId);
        if (roof?.polygon) roof.polygon.material = Color.fromCssColorString(on ? ORANGE : ROOF) as never; // opaque: see the note where these are created
      }
    });
    scene.requestRender();
  };

  // ---- item markers: lost (orange L), found (ink F), collect (ink pin), the spot being chosen (orange target)
  const marks = new Map<string, Entity>();
  const tagIcon = (kind: Marker["kind"]) => {
    const fill = kind === "lost" || kind === "pick" ? ORANGE : "#0B0B0B";
    const glyph = kind === "lost" ? "L" : kind === "found" ? "F" : kind === "collect" ? "★" : "";
    const text = kind === "lost" || kind === "pick" ? "#0B0B0B" : "#FFFFFF";
    const inner = kind === "pick" ? `<circle cx="20" cy="19" r="6" fill="#FFFFFF"/>` : `<text x="20" y="24" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" font-weight="700" fill="${text}">${glyph}</text>`;
    const svg =
      `<svg xmlns="http://www.w3.org/2000/svg" width="40" height="48" viewBox="0 0 40 48"><path d="M20 46C20 46 4 30 4 19a16 16 0 0 1 32 0c0 11-16 27-16 27z" fill="${fill}" stroke="#FFFFFF" stroke-width="3"/>${inner}</svg>`;
    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  };
  const setMarkers = (list: Marker[]) => {
    marks.forEach((e) => viewer.entities.remove(e));
    marks.clear();
    for (const m of list) {
      marks.set(
        m.id,
        viewer.entities.add({
          position: Cartesian3.fromDegrees(m.lon, m.lat, useIon || useGoogle ? 0 : ground + 4),
          billboard: {
            image: tagIcon(m.kind),
            width: m.kind === "pick" ? 40 : 32,
            height: m.kind === "pick" ? 48 : 38,
            verticalOrigin: VerticalOrigin.BOTTOM,
            heightReference: clamp,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
          label: m.label
            ? {
                text: m.label,
                font: "600 12px 'IBM Plex Sans', Helvetica, Arial, sans-serif",
                style: LabelStyle.FILL,
                fillColor: Color.fromCssColorString("#0B0B0B"),
                showBackground: true,
                backgroundColor: Color.WHITE.withAlpha(0.94),
                backgroundPadding: new Cartesian2(7, 4),
                horizontalOrigin: HorizontalOrigin.LEFT,
                verticalOrigin: VerticalOrigin.BOTTOM,
                pixelOffset: new Cartesian2(18, -12),
                heightReference: clamp,
                distanceDisplayCondition: new DistanceDisplayCondition(0, 600),
                disableDepthTestDistance: Number.POSITIVE_INFINITY,
              }
            : undefined,
        }),
      );
    }
    scene.requestRender();
  };

  if (useGoogle) {
    createGooglePhotorealistic3DTileset({ key: googleKey, onlyUsingWithGoogleGeocoder: true })
      .then(async (tiles) => {
        if (viewer.isDestroyed()) return;
        scene.primitives.add(tiles);
        scene.globe.show = false;
        for (const e of roofs.values()) e.show = false;
        // The tiles sit at their true height; aim the camera at the ground there, not at height 0.
        const [c] = scene.sampleHeightSupported
          ? await scene.sampleHeightMostDetailed([Cartographic.fromDegrees(campus.lon, campus.lat)]).catch(() => [])
          : [];
        settleGround(c?.height);
      })
      .catch((err) => {
        console.warn("Google 3D tiles unavailable, falling back to flat footprints", err);
        buildFlatBoxes();
      });
  }

  return {
    setMarkers,
    focus(lat, lon, range = 240) {
      frame(new BoundingSphere(Cartesian3.fromDegrees(lon, lat, ground), 30), camera.heading || HEADING, PITCH_3D, range, 1.1);
    },
    select(id) {
      selectedId = id;
      paint();
      const p = pickupPoints.find((q) => q.id === id);
      if (p) frame(new BoundingSphere(Cartesian3.fromDegrees(p.lon, p.lat, 8), 30), camera.heading, nowPitch() === PITCH_TOP ? PITCH_TOP : PITCH_3D, 230, 1.3);
    },
    pitch(mode) {
      frame(new BoundingSphere(centre, campus.radius), camera.heading, mode === "top" ? PITCH_TOP : PITCH_3D, viewRange, 1.1);
    },
    turn() {
      frame(new BoundingSphere(centre, campus.radius), camera.heading + CesiumMath.PI_OVER_TWO, nowPitch(), viewRange, 1.4);
    },
    reset() {
      selectedId = null;
      paint();
      frame(sphere, HEADING, PITCH_3D, viewRange, 1.4);
    },
    destroy() {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("contextmenu", noMenu);
      if (!viewer.isDestroyed()) viewer.destroy();
    },
  };
}
