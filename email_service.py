"""
email_service.py
Sends present/absent HTML emails to students via SMTP (async).
"""

import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_FROM_NAME, SCHOOL_NAME

logger = logging.getLogger(__name__)

# ── HTML Templates ────────────────────────────────────────────────────────────

_BASE = """
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:520px;margin:0 auto;background:#fff;border-radius:10px;overflow:hidden;border:1px solid #e8e8e8">
  <div style="background:{header_bg};padding:28px 32px">
    <p style="color:rgba(255,255,255,.7);margin:0 0 4px;font-size:13px">{school}</p>
    <h2 style="color:#fff;margin:0;font-size:20px;font-weight:500">{headline}</h2>
  </div>
  <div style="padding:28px 32px">
    <p style="font-size:15px;color:#111;margin:0 0 22px">Hi <strong>{student_name}</strong>, {body_intro}</p>
    {details}
    {extra}
  </div>
  <div style="padding:16px 32px;background:#f7f7f7;border-top:1px solid #eee;font-size:12px;color:#999">
    Automated message from {school} SmartAttend · Do not reply
  </div>
</div>
"""

_ROW = '<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #f0f0f0;font-size:14px"><span style="color:#666">{k}</span><span style="font-weight:500;color:#111">{v}</span></div>'

_WARN = '<div style="margin-top:20px;padding:14px 16px;background:#FAEEDA;border-left:3px solid #BA7517;border-radius:0 6px 6px 0;font-size:13px;color:#633806">⚠️ {msg}</div>'
_GOOD = '<div style="margin-top:20px;padding:14px 16px;background:#EAF3DE;border-left:3px solid #1D9E75;border-radius:0 6px 6px 0;font-size:13px;color:#085041">✓ {msg}</div>'


def _build_html(student_name, status, session_date, subject, class_name,
                confidence_pct, attendance_percent):
    is_present = status == "present"
    rows = "".join([
        _ROW.format(k="Date", v=session_date),
        _ROW.format(k="Subject", v=subject or "—"),
        _ROW.format(k="Class", v=class_name),
    ])
    if is_present and confidence_pct:
        rows += _ROW.format(k="Recognition confidence", v=f"{confidence_pct}%")

    extra = ""
    if attendance_percent is not None:
        pct_str = f"{attendance_percent}%"
        rows += _ROW.format(k="Overall attendance", v=pct_str)
        if not is_present and attendance_percent < 75:
            extra = _WARN.format(msg="Attendance has dropped below 75%. Please contact your faculty advisor.")
        elif is_present and attendance_percent >= 85:
            extra = _GOOD.format(msg=f"Great work — your attendance is {pct_str} this semester!")

    return _BASE.format(
        header_bg="#0F6E56" if is_present else "#A32D2D",
        school=SCHOOL_NAME,
        headline="✓ Attendance Marked — Present" if is_present else "✗ Marked Absent",
        student_name=student_name,
        body_intro="your attendance has been recorded for today's session." if is_present
                   else "you were marked <strong>absent</strong> from today's session.",
        details=rows,
        extra=extra,
    )


# ── Sending ───────────────────────────────────────────────────────────────────

def send_email(to_email: str, subject: str, html: str) -> bool:
    """Send one email via SMTP. Returns True on success."""
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("SMTP credentials not configured — skipping email")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{EMAIL_FROM_NAME} <{EMAIL_FROM or SMTP_USER}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [to_email], msg.as_string())
        return True
    except Exception as e:
        logger.error(f"Email failed to {to_email}: {e}")
        return False


def send_attendance_emails(
    records: list[dict],
    session_info: dict,
) -> tuple[int, int]:
    """
    Send present/absent emails to all students in a session.
    Returns (sent_count, failed_count).
    """
    sent = failed = 0
    for rec in records:
        if not rec.get("email"):
            failed += 1
            continue

        is_present = rec["status"] == "present"
        conf_pct   = round(rec["confidence"] * 100) if rec.get("confidence") else None
        att_pct    = rec.get("attendance_percent")

        html = _build_html(
            student_name=rec.get("name", "Student"),
            status=rec["status"],
            session_date=session_info.get("session_date", ""),
            subject=session_info.get("subject", "Class"),
            class_name=session_info.get("class_name", ""),
            confidence_pct=conf_pct,
            attendance_percent=att_pct,
        )
        subject_line = (
            f"✓ Present — {session_info.get('subject','Class')} · {session_info.get('session_date','')}"
            if is_present else
            f"⚠ Absent — {session_info.get('subject','Class')} · {session_info.get('session_date','')}"
        )
        ok = send_email(rec["email"], subject_line, html)
        if ok:
            sent += 1
        else:
            failed += 1

    return sent, failed
