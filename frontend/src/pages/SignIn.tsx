import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, Navigate, useNavigate, useSearchParams } from "react-router-dom";
import { api, ApiError } from "../api";
import { useAuth } from "../auth";
import { useT } from "../i18n";
import { Icon, IconSprite } from "../components/Icons";
import Brand from "../components/Brand";
import ThemeToggle from "../components/ThemeToggle";

const RESEND_S = 30;

// Demo accounts, seeded by scripts/seed_demo.py. Shown only when the backend is in demo mode.
const DEMO_ACCOUNTS = [
  { email: "asha@vnrvjiet.ac.in", name: "Asha", note: "Student with a matched laptop" },
  { email: "rahul@vnrvjiet.ac.in", name: "Rahul", note: "Finder who turned items in" },
  { email: "riya@vnrvjiet.ac.in", name: "Riya", note: "Has a disputed claim" },
  { email: "desk@vnrvjiet.ac.in", name: "Desk", note: "Reviews claims, confirms handovers" },
  { email: "admin@vnrvjiet.ac.in", name: "Admin", note: "Insights and evaluation" },
];

/** College email in, six-digit code out. No passwords. */
export default function SignIn() {
  const { t } = useT();
  const { user, signIn } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const asked = params.get("then") ?? "";
  // Only same-site paths: a leading slash, and not "//host".
  const then = asked.startsWith("/") && !asked.startsWith("//") ? asked : "/home";

  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [wait, setWait] = useState(0);
  const [demo, setDemo] = useState<string | null>(null);
  const [waking, setWaking] = useState(false);
  const codeEl = useRef<HTMLInputElement>(null);

  // Demo mode: the backend says which code it will accept, so the page can show it. A free host puts the server to
  // sleep when idle, so the first tries can fail: keep asking for a while instead of giving up after one miss.
  useEffect(() => {
    let dead = false;
    (async () => {
      for (let i = 0; i < 10 && !dead; i++) {
        try {
          const c = await api.authConfig();
          if (!dead) { setDemo(c.demo_otp); setWaking(false); }
          return;
        } catch {
          if (!dead) setWaking(true);
          await new Promise((r) => setTimeout(r, 4000));
        }
      }
      if (!dead) setWaking(false);
    })();
    return () => { dead = true; };
  }, []);

  useEffect(() => {
    if (wait <= 0) return;
    const id = window.setTimeout(() => setWait((w) => w - 1), 1000);
    return () => clearTimeout(id);
  }, [wait]);
  useEffect(() => { if (sent) codeEl.current?.focus(); }, [sent]);

  if (user) return <Navigate to={then} replace />;

  async function send(e?: FormEvent) {
    e?.preventDefault();
    setBusy(true);
    setErr("");
    try {
      await api.requestCode(email.trim());
      setSent(true);
      setWait(RESEND_S);
    } catch (x) {
      setErr((x as ApiError).message);
    } finally {
      setBusy(false);
    }
  }

  async function quick(addr: string) {
    setBusy(true);
    setErr("");
    try {
      await api.requestCode(addr);
      const r = await api.verifyCode(addr, demo ?? "");
      signIn(r.token, r.user);
      navigate(then, { replace: true });
    } catch (x) {
      setErr((x as ApiError).message);
    } finally {
      setBusy(false);
    }
  }

  async function verify(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      const r = await api.verifyCode(email.trim(), code.trim());
      signIn(r.token, r.user);
      navigate(then, { replace: true });
    } catch (x) {
      setErr((x as ApiError).message);
      setCode("");
      codeEl.current?.focus();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth">
      <IconSprite />
      <header className="auth__bar">
        <Link className="brand" to="/welcome" aria-label="reunite, home">
          <Brand />
        </Link>
        <ThemeToggle />
      </header>

      <main className="auth__main" id="main">
        <div className="auth__card">
          <div>
            <p className="eyebrow"><Icon name="star" size={12} /> Sign in</p>
            <h1>Welcome <em>back.</em></h1>
          </div>

          {!sent ? (
            <form className="auth__form" onSubmit={send} noValidate>
              <p className="lede">{t("signin.title")}. {demo ? "This is a demo: no email is sent, and the code appears on the next step." : "We'll email you a six-digit code, so there is no password to remember."}</p>
              <label className="field">
                <span className="field__label">{t("signin.email")}</span>
                <input
                  className="input" type="email" inputMode="email" autoComplete="email" autoFocus required
                  value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@vnrvjiet.ac.in"
                  aria-invalid={!!err} aria-describedby={err ? "auth-err" : undefined}
                />
                {err && <span className="field__error" id="auth-err" role="alert">{err}</span>}
              </label>
              <button className="btn btn--lost btn--block" disabled={busy || !email.includes("@")} aria-busy={busy}>
                {t("signin.send")} <Icon name="up-right" size={18} />
              </button>
            </form>
          ) : (
            <form className="auth__form" onSubmit={verify}>
              {demo ? (
                <p className="auth__demo" role="note">Demo mode: no email is sent. Enter <strong className="mono">{demo}</strong>.</p>
              ) : (
                <p className="auth__sent">{t("signin.sent")} <strong>{email}</strong></p>
              )}
              <label className="field">
                <span className="field__label">{t("signin.code")}</span>
                <input
                  ref={codeEl} className="input input--code" inputMode="numeric" autoComplete="one-time-code" maxLength={6} required
                  value={code} onChange={(e) => setCode(e.target.value.replace(/\D/g, ""))} placeholder="······"
                  aria-invalid={!!err} aria-describedby={err ? "auth-err" : undefined}
                />
                {err && <span className="field__error" id="auth-err" role="alert">{err}</span>}
              </label>
              <button className="btn btn--lost btn--block" disabled={busy || code.length !== 6} aria-busy={busy}>
                {t("signin.verify")} <Icon name="up-right" size={18} />
              </button>
              <div className="auth__links">
                <button type="button" className="linkbtn" onClick={() => { setSent(false); setCode(""); setErr(""); }}>{t("signin.change")}</button>
                <button type="button" className="linkbtn" disabled={wait > 0 || busy} onClick={() => void send()}>
                  {wait > 0 ? `Resend in ${wait}s` : "Resend code"}
                </button>
              </div>
            </form>
          )}

          {waking && !demo && !sent && (
            <p className="auth__note" role="status">Waking the server. The demo accounts appear here in a moment, which can take up to a minute.</p>
          )}

          {demo && !sent && (
            <div className="demo">
              <span className="field__label">Demo accounts, one click</span>
              <ul className="demo__list" role="list">
                {DEMO_ACCOUNTS.map((a) => (
                  <li key={a.email}>
                    <button type="button" className="demo__btn" disabled={busy} onClick={() => quick(a.email)}>
                      <strong>{a.name}</strong>
                      <span>{a.note}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <p className="auth__note">Items only. Your email is used to sign you in and to tell you when something matches.</p>
          <Link className="textlink" to="/welcome"><Icon name="left" /> Back to the globe</Link>
        </div>
      </main>
    </div>
  );
}
