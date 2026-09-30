import os
import tempfile
import unittest

try:
    import flask  # noqa: F401
    import cv2    # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "requiere flask y opencv")
class DashboardAuthTests(unittest.TestCase):
    def setUp(self):
        from config.settings import Config
        import dashboard.app as dash
        self.dash, self.Config = dash, Config
        self.tmp = tempfile.mkdtemp()
        Config.DATA_DIR = self.tmp
        Config.DB_PATH = os.path.join(self.tmp, "events.db")
        Config.CAPTURES_DIR = os.path.join(self.tmp, "captures")
        os.makedirs(Config.CAPTURES_DIR)
        Config.DASHBOARD_PASSWORD = "clave-de-prueba"
        dash._failed.clear()
        dash.configure_auth()
        self.client = dash.app.test_client()

    def login(self, pw="clave-de-prueba"):
        return self.client.post("/login", data={"password": pw})

    def test_everything_requires_login(self):
        self.assertEqual(self.client.get("/").status_code, 302)
        for path in ("/api/events", "/api/status", "/video_feed", "/captures/x.jpg"):
            self.assertEqual(self.client.get(path).status_code, 401, path)

    def test_wrong_password_rejected(self):
        self.assertEqual(self.login("mala").status_code, 401)
        self.assertEqual(self.client.get("/api/status").status_code, 401)

    def test_correct_password_grants_access_and_logout_revokes(self):
        self.assertEqual(self.login().status_code, 302)
        self.assertEqual(self.client.get("/api/status").status_code, 200)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.client.post("/logout")
        self.assertEqual(self.client.get("/api/status").status_code, 401)

    def test_lockout_after_repeated_failures(self):
        for _ in range(self.dash.MAX_FAILED_LOGINS):
            self.assertEqual(self.login("mala").status_code, 401)
        self.assertEqual(self.login().status_code, 429)   # ni siquiera la buena entra mientras dure el bloqueo

    def test_path_traversal_blocked(self):
        self.login()
        self.assertEqual(self.client.get("/captures/..%2Fevents.db").status_code, 404)

    def test_security_headers(self):
        r = self.client.get("/login")
        self.assertEqual(r.headers["X-Frame-Options"], "DENY")
        self.assertIn("default-src 'self'", r.headers["Content-Security-Policy"])

    def test_unconfigured_password_serves_nothing(self):
        self.dash.app.config["SENTINEL_PASSWORD"] = None
        self.assertEqual(self.client.get("/api/status").status_code, 503)
        self.assertEqual(self.client.get("/").status_code, 503)

    def test_generated_password_is_persisted(self):
        self.Config.DASHBOARD_PASSWORD = ""
        os.remove(os.path.join(self.tmp, "dashboard_password.txt")) if os.path.exists(
            os.path.join(self.tmp, "dashboard_password.txt")) else None
        pw1, auto1, _ = self.dash.configure_auth()
        pw2, auto2, _ = self.dash.configure_auth()
        self.assertTrue(auto1 and auto2)
        self.assertEqual(pw1, pw2)
        self.assertGreaterEqual(len(pw1), 12)


if __name__ == "__main__":
    unittest.main()
