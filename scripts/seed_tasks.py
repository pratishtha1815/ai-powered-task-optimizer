"""
scripts/seed_tasks.py
─────────────────────────────────────────────────────────────
One-time script that generates data/tasks.csv with 60 realistic
tasks spread across three cognitive-load tiers.

Cognitive Load Tiers
────────────────────
  LOW    (1–3)  : repetitive, administrative, minimal focus required
  MEDIUM (4–6)  : standard engineering / analytical work
  HIGH   (7–10) : creative, complex problem-solving, leadership tasks

Usage:
    python scripts/seed_tasks.py

Output:
    data/tasks.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from pathlib import Path
from config import DATA_DIR
from loguru import logger

OUTPUT_PATH = DATA_DIR / "tasks.csv"


# ─────────────────────────────────────────────────────────────
# Task catalogue
# Each dict: id, title, description, category, cognitive_load,
#            estimated_hours, tags
# ─────────────────────────────────────────────────────────────
TASKS = [
    # ── LOW cognitive load (1–3) ─────────────────────────────
    {
        "id": 1, "title": "Organise email inbox",
        "description": "Sort, label, and archive emails from the past week.",
        "category": "Administrative", "cognitive_load": 1,
        "estimated_hours": 0.5,
        "tags": "admin,email,organisation"
    },
    {
        "id": 2, "title": "Update meeting notes",
        "description": "Transcribe and format notes from yesterday's standup.",
        "category": "Administrative", "cognitive_load": 1,
        "estimated_hours": 0.5,
        "tags": "admin,documentation,notes"
    },
    {
        "id": 3, "title": "File expense reports",
        "description": "Submit last month's expense receipts through the HR portal.",
        "category": "Administrative", "cognitive_load": 1,
        "estimated_hours": 0.5,
        "tags": "admin,finance,hr"
    },
    {
        "id": 4, "title": "Update team calendar",
        "description": "Add upcoming deadlines and review cycles to the shared calendar.",
        "category": "Administrative", "cognitive_load": 1,
        "estimated_hours": 0.5,
        "tags": "admin,scheduling,team"
    },
    {
        "id": 5, "title": "Respond to routine emails",
        "description": "Reply to low-priority internal emails and status requests.",
        "category": "Communication", "cognitive_load": 2,
        "estimated_hours": 1.0,
        "tags": "communication,email,low-stress"
    },
    {
        "id": 6, "title": "Review and approve timesheets",
        "description": "Check team timesheets for the week and approve in the system.",
        "category": "Administrative", "cognitive_load": 2,
        "estimated_hours": 0.5,
        "tags": "admin,hr,management"
    },
    {
        "id": 7, "title": "Read industry newsletter",
        "description": "Browse this week's tech digest and bookmark relevant articles.",
        "category": "Learning", "cognitive_load": 2,
        "estimated_hours": 0.5,
        "tags": "learning,research,passive"
    },
    {
        "id": 8, "title": "Tidy project documentation folder",
        "description": "Remove duplicate files and rename documents to convention.",
        "category": "Administrative", "cognitive_load": 2,
        "estimated_hours": 1.0,
        "tags": "admin,documentation,organisation"
    },
    {
        "id": 9, "title": "Watch recorded training video",
        "description": "Complete the 45-minute onboarding video for the new CI/CD tool.",
        "category": "Learning", "cognitive_load": 2,
        "estimated_hours": 1.0,
        "tags": "learning,training,passive"
    },
    {
        "id": 10, "title": "Update personal task tracker",
        "description": "Review open tickets and ensure statuses are current in Jira.",
        "category": "Administrative", "cognitive_load": 2,
        "estimated_hours": 0.5,
        "tags": "admin,planning,organisation"
    },
    {
        "id": 11, "title": "Write daily stand-up update",
        "description": "Compose a brief update for the async stand-up channel.",
        "category": "Communication", "cognitive_load": 2,
        "estimated_hours": 0.25,
        "tags": "communication,team,async"
    },
    {
        "id": 12, "title": "Review pull request comments",
        "description": "Read through open PR comments without responding — note action items.",
        "category": "Engineering", "cognitive_load": 3,
        "estimated_hours": 1.0,
        "tags": "engineering,code-review,low-stress"
    },
    {
        "id": 13, "title": "Prepare meeting agenda",
        "description": "Draft discussion points for the upcoming sprint planning session.",
        "category": "Administrative", "cognitive_load": 3,
        "estimated_hours": 0.5,
        "tags": "admin,planning,meetings"
    },
    {
        "id": 14, "title": "Update README file",
        "description": "Refresh the project README with recent environment changes.",
        "category": "Engineering", "cognitive_load": 3,
        "estimated_hours": 1.0,
        "tags": "engineering,documentation,writing"
    },
    {
        "id": 15, "title": "Organise Slack channels",
        "description": "Archive stale channels and pin key resources in active ones.",
        "category": "Administrative", "cognitive_load": 3,
        "estimated_hours": 0.5,
        "tags": "admin,communication,team"
    },
    {
        "id": 16, "title": "Re-run test suite on existing branch",
        "description": "Execute unit tests on a known-good branch to verify green state.",
        "category": "Engineering", "cognitive_load": 3,
        "estimated_hours": 0.5,
        "tags": "engineering,testing,validation"
    },
    {
        "id": 17, "title": "Review closed tickets for patterns",
        "description": "Scan last sprint's resolved tickets to identify recurring issues.",
        "category": "Analysis", "cognitive_load": 3,
        "estimated_hours": 1.0,
        "tags": "analysis,retrospective,low-stress"
    },
    {
        "id": 18, "title": "Create or update team wiki page",
        "description": "Add a how-to guide for a recently learned tool or process.",
        "category": "Documentation", "cognitive_load": 3,
        "estimated_hours": 1.5,
        "tags": "documentation,writing,knowledge-sharing"
    },
    {
        "id": 19, "title": "Attend team social / coffee chat",
        "description": "Join the informal 15-minute virtual team coffee session.",
        "category": "Well-being", "cognitive_load": 1,
        "estimated_hours": 0.25,
        "tags": "well-being,social,team"
    },
    {
        "id": 20, "title": "Take a focused walk break",
        "description": "Step away from screens for a 10-minute mindful walk.",
        "category": "Well-being", "cognitive_load": 1,
        "estimated_hours": 0.17,
        "tags": "well-being,self-care,break"
    },

    # ── MEDIUM cognitive load (4–6) ──────────────────────────
    {
        "id": 21, "title": "Write unit tests for existing feature",
        "description": "Add Jest/pytest tests for the authentication module.",
        "category": "Engineering", "cognitive_load": 4,
        "estimated_hours": 2.0,
        "tags": "engineering,testing,quality"
    },
    {
        "id": 22, "title": "Fix medium-priority bug",
        "description": "Resolve a reported layout bug in the dashboard table component.",
        "category": "Engineering", "cognitive_load": 4,
        "estimated_hours": 2.0,
        "tags": "engineering,bugfix,debugging"
    },
    {
        "id": 23, "title": "Refactor a small module",
        "description": "Improve readability of the user-preferences helper module.",
        "category": "Engineering", "cognitive_load": 5,
        "estimated_hours": 2.0,
        "tags": "engineering,refactoring,code-quality"
    },
    {
        "id": 24, "title": "Conduct peer code review",
        "description": "Review and leave constructive comments on two open pull requests.",
        "category": "Engineering", "cognitive_load": 5,
        "estimated_hours": 1.5,
        "tags": "engineering,code-review,collaboration"
    },
    {
        "id": 25, "title": "Analyse sprint velocity data",
        "description": "Pull last four sprints' velocity metrics and create a summary chart.",
        "category": "Analysis", "cognitive_load": 5,
        "estimated_hours": 2.0,
        "tags": "analysis,metrics,reporting"
    },
    {
        "id": 26, "title": "Implement API endpoint",
        "description": "Build the GET /users/{id}/preferences endpoint per the spec.",
        "category": "Engineering", "cognitive_load": 5,
        "estimated_hours": 3.0,
        "tags": "engineering,backend,api"
    },
    {
        "id": 27, "title": "Write technical blog post",
        "description": "Draft a 600-word internal article on the new deployment pipeline.",
        "category": "Communication", "cognitive_load": 5,
        "estimated_hours": 2.5,
        "tags": "writing,knowledge-sharing,communication"
    },
    {
        "id": 28, "title": "Create data visualisation dashboard",
        "description": "Build a Plotly chart showing user sign-up trends for Q2.",
        "category": "Analytics", "cognitive_load": 5,
        "estimated_hours": 3.0,
        "tags": "analytics,visualisation,reporting"
    },
    {
        "id": 29, "title": "Database schema migration",
        "description": "Write and test the Alembic migration for the new orders table.",
        "category": "Engineering", "cognitive_load": 6,
        "estimated_hours": 3.0,
        "tags": "engineering,database,migration"
    },
    {
        "id": 30, "title": "Integrate third-party API",
        "description": "Connect the payment gateway sandbox and test webhook events.",
        "category": "Engineering", "cognitive_load": 6,
        "estimated_hours": 4.0,
        "tags": "engineering,integration,api"
    },
    {
        "id": 31, "title": "Lead sprint retrospective",
        "description": "Facilitate the team retrospective and document key action items.",
        "category": "Leadership", "cognitive_load": 4,
        "estimated_hours": 1.5,
        "tags": "leadership,agile,facilitation"
    },
    {
        "id": 32, "title": "Create onboarding checklist",
        "description": "Draft a step-by-step onboarding guide for new team members.",
        "category": "Documentation", "cognitive_load": 4,
        "estimated_hours": 2.0,
        "tags": "documentation,hr,knowledge-sharing"
    },
    {
        "id": 33, "title": "Perform security patch updates",
        "description": "Apply pending OS and dependency security patches to staging.",
        "category": "Engineering", "cognitive_load": 4,
        "estimated_hours": 2.0,
        "tags": "engineering,security,devops"
    },
    {
        "id": 34, "title": "Research competitor features",
        "description": "Explore two competitor products and note three differentiating features.",
        "category": "Research", "cognitive_load": 4,
        "estimated_hours": 2.0,
        "tags": "research,product,analysis"
    },
    {
        "id": 35, "title": "Prepare weekly status report",
        "description": "Compile team progress, blockers, and next-week goals into a report.",
        "category": "Reporting", "cognitive_load": 4,
        "estimated_hours": 1.5,
        "tags": "reporting,management,communication"
    },
    {
        "id": 36, "title": "Set up monitoring alerts",
        "description": "Configure Grafana / CloudWatch alerts for API error rate thresholds.",
        "category": "DevOps", "cognitive_load": 5,
        "estimated_hours": 2.5,
        "tags": "devops,monitoring,infrastructure"
    },
    {
        "id": 37, "title": "Conduct user interview",
        "description": "Run a 30-minute structured UX interview with a power user.",
        "category": "Research", "cognitive_load": 5,
        "estimated_hours": 1.5,
        "tags": "research,ux,product"
    },
    {
        "id": 38, "title": "Update CI/CD pipeline configuration",
        "description": "Add linting and test coverage steps to the GitHub Actions workflow.",
        "category": "DevOps", "cognitive_load": 5,
        "estimated_hours": 2.0,
        "tags": "devops,ci-cd,automation"
    },
    {
        "id": 39, "title": "Write project risk register",
        "description": "Identify top-5 risks for the current quarter and propose mitigations.",
        "category": "Management", "cognitive_load": 6,
        "estimated_hours": 2.0,
        "tags": "management,planning,risk"
    },
    {
        "id": 40, "title": "Implement caching layer",
        "description": "Add Redis caching to the product catalogue endpoint.",
        "category": "Engineering", "cognitive_load": 6,
        "estimated_hours": 4.0,
        "tags": "engineering,performance,backend"
    },

    # ── HIGH cognitive load (7–10) ───────────────────────────
    {
        "id": 41, "title": "Design new system architecture",
        "description": "Produce a C4 architecture diagram for the microservices refactor.",
        "category": "Architecture", "cognitive_load": 7,
        "estimated_hours": 4.0,
        "tags": "architecture,design,complex"
    },
    {
        "id": 42, "title": "Debug production performance issue",
        "description": "Profile and resolve the P0 latency spike reported in checkout flow.",
        "category": "Engineering", "cognitive_load": 8,
        "estimated_hours": 4.0,
        "tags": "engineering,debugging,production,complex"
    },
    {
        "id": 43, "title": "Build machine learning pipeline",
        "description": "Design and implement an end-to-end feature engineering pipeline.",
        "category": "ML/AI", "cognitive_load": 9,
        "estimated_hours": 6.0,
        "tags": "ml,ai,data-science,complex"
    },
    {
        "id": 44, "title": "Lead technical interview",
        "description": "Conduct a two-hour live coding and system design interview session.",
        "category": "Leadership", "cognitive_load": 7,
        "estimated_hours": 2.5,
        "tags": "leadership,hiring,technical"
    },
    {
        "id": 45, "title": "Architect distributed caching strategy",
        "description": "Design a multi-region Redis cluster strategy with failover planning.",
        "category": "Architecture", "cognitive_load": 9,
        "estimated_hours": 5.0,
        "tags": "architecture,distributed-systems,complex"
    },
    {
        "id": 46, "title": "Write engineering RFC",
        "description": "Draft a Request for Comments on migrating to event-driven architecture.",
        "category": "Engineering", "cognitive_load": 8,
        "estimated_hours": 5.0,
        "tags": "engineering,architecture,writing,leadership"
    },
    {
        "id": 47, "title": "Resolve critical security vulnerability",
        "description": "Investigate the reported SQL injection vector and deploy a fix.",
        "category": "Engineering", "cognitive_load": 9,
        "estimated_hours": 6.0,
        "tags": "engineering,security,critical,complex"
    },
    {
        "id": 48, "title": "Define OKRs for next quarter",
        "description": "Facilitate team OKR planning session and write final measurable goals.",
        "category": "Leadership", "cognitive_load": 7,
        "estimated_hours": 3.0,
        "tags": "leadership,strategy,planning"
    },
    {
        "id": 49, "title": "Prototype new product feature",
        "description": "Build a functional MVP of the AI-driven search feature from scratch.",
        "category": "Engineering", "cognitive_load": 8,
        "estimated_hours": 6.0,
        "tags": "engineering,innovation,product,complex"
    },
    {
        "id": 50, "title": "Present roadmap to stakeholders",
        "description": "Prepare and deliver a 45-minute quarterly roadmap presentation.",
        "category": "Leadership", "cognitive_load": 7,
        "estimated_hours": 3.0,
        "tags": "leadership,communication,presentation"
    },
    {
        "id": 51, "title": "Implement real-time data pipeline",
        "description": "Build a Kafka consumer pipeline for streaming user event data.",
        "category": "Engineering", "cognitive_load": 9,
        "estimated_hours": 8.0,
        "tags": "engineering,data,streaming,complex"
    },
    {
        "id": 52, "title": "Create ML model evaluation framework",
        "description": "Build reproducible evaluation harness with cross-validation and bias tests.",
        "category": "ML/AI", "cognitive_load": 8,
        "estimated_hours": 5.0,
        "tags": "ml,ai,evaluation,quality"
    },
    {
        "id": 53, "title": "Negotiate vendor contract",
        "description": "Evaluate two SaaS vendor proposals and draft negotiation terms.",
        "category": "Management", "cognitive_load": 7,
        "estimated_hours": 3.0,
        "tags": "management,procurement,negotiation"
    },
    {
        "id": 54, "title": "Plan and run hackathon",
        "description": "Organise a 24-hour internal hackathon: logistics, judging and comms.",
        "category": "Leadership", "cognitive_load": 7,
        "estimated_hours": 8.0,
        "tags": "leadership,event,innovation,team"
    },
    {
        "id": 55, "title": "Implement end-to-end encryption",
        "description": "Design and integrate E2E encryption for user messages using libsodium.",
        "category": "Engineering", "cognitive_load": 10,
        "estimated_hours": 8.0,
        "tags": "engineering,security,complex,cryptography"
    },
    {
        "id": 56, "title": "Build graph neural network model",
        "description": "Research and implement a GNN for the social recommendation engine.",
        "category": "ML/AI", "cognitive_load": 10,
        "estimated_hours": 10.0,
        "tags": "ml,ai,research,complex"
    },
    {
        "id": 57, "title": "Design capacity planning model",
        "description": "Build a financial and headcount model for next-year team growth.",
        "category": "Management", "cognitive_load": 8,
        "estimated_hours": 6.0,
        "tags": "management,planning,finance,strategy"
    },
    {
        "id": 58, "title": "Create accessibility audit and fix plan",
        "description": "Run WCAG 2.1 AA audit and prioritise top-10 remediation items.",
        "category": "Engineering", "cognitive_load": 7,
        "estimated_hours": 4.0,
        "tags": "engineering,accessibility,quality,complex"
    },
    {
        "id": 59, "title": "Architect multi-tenant SaaS solution",
        "description": "Design the tenant isolation, billing and data strategy for SaaS launch.",
        "category": "Architecture", "cognitive_load": 10,
        "estimated_hours": 8.0,
        "tags": "architecture,saas,strategy,complex"
    },
    {
        "id": 60, "title": "Build automated regression test suite",
        "description": "Create a Selenium / Playwright full-regression suite for the web app.",
        "category": "Engineering", "cognitive_load": 7,
        "estimated_hours": 5.0,
        "tags": "engineering,testing,automation,quality"
    },
]


def seed_tasks(output_path: Path = OUTPUT_PATH) -> None:
    df = pd.DataFrame(TASKS)

    # Derive tier from cognitive_load
    def _tier(cl: int) -> str:
        if cl <= 3: return "Low"
        if cl <= 6: return "Medium"
        return "High"

    df["tier"] = df["cognitive_load"].apply(_tier)
    df.to_csv(output_path, index=False)

    counts = df["tier"].value_counts().to_dict()
    logger.info(
        f"Seeded {len(df)} tasks to {output_path} | "
        f"Low={counts.get('Low',0)}, "
        f"Medium={counts.get('Medium',0)}, "
        f"High={counts.get('High',0)}"
    )
    print(f"[OK]  {len(df)} tasks written -> {output_path}")
    print(f"   Low: {counts.get('Low',0)} | "
          f"Medium: {counts.get('Medium',0)} | "
          f"High: {counts.get('High',0)}")


if __name__ == "__main__":
    seed_tasks()
