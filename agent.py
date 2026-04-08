import re
import pypdf
import io
from llm_provider import get_llm, get_embeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import os
import json
import ast

llm = get_llm()

# ------------------------------------------------------------------
# Rule-based weakness suggestions
# flan-t5-base produces nonsense for weakness analysis
# (e.g. "make sure resume is not an HTML resume", "Russian flag")
# These hand-crafted suggestions are accurate and always useful.
# ------------------------------------------------------------------
SKILL_SUGGESTIONS = {
    # Frontend
    "react":        ["Add React projects with component architecture details.",
                     "Mention React hooks (useState, useEffect, useContext) explicitly.",
                     "Include React with TypeScript or Redux in project descriptions."],
    "vue":          ["Add a Vue.js project with Vuex or Vue Router.",
                     "Mention Vue 3 Composition API or Options API experience.",
                     "Include Vue with TypeScript in your skills section."],
    "angular":      ["Add an Angular project using Angular CLI and RxJS.",
                     "Mention Angular modules, services, and dependency injection.",
                     "Include Angular version (e.g. Angular 15+) in your skills."],
    "html":         ["Add 'HTML5' explicitly to your skills and project tech stacks.",
                     "Mention semantic HTML, accessibility (ARIA), and SEO practices.",
                     "Include HTML5 in project descriptions alongside CSS and JS."],
    "html5":        ["Add 'HTML5' explicitly to your skills and project tech stacks.",
                     "Mention semantic HTML5 elements (header, main, article, section).",
                     "Include HTML5 canvas or Web APIs if used in projects."],
    "css":          ["Add CSS3, Flexbox, and Grid to your skills section explicitly.",
                     "Mention CSS frameworks (Tailwind, Bootstrap) in project stacks.",
                     "Include responsive design and mobile-first CSS experience."],
    "css3":         ["Add 'CSS3' explicitly alongside HTML5 in your skills.",
                     "Mention CSS animations, transitions, and custom properties.",
                     "Include CSS preprocessors (SASS/SCSS) if applicable."],
    "javascript":   ["Add JavaScript (ES6+) explicitly to your technical skills.",
                     "Mention async/await, Promises, closures, and ES6 features.",
                     "Include vanilla JavaScript projects alongside framework work."],
    "typescript":   ["Add TypeScript to your skills and specify version or usage.",
                     "Mention TypeScript interfaces, generics, and strict mode.",
                     "Migrate a project to TypeScript and describe it in experience."],
    "next.js":      ["Add a Next.js project with SSR or SSG implementation details.",
                     "Mention Next.js routing, API routes, and Image optimization.",
                     "Include Next.js with TypeScript and Tailwind in project stack."],
    "svelte":       ["Add a Svelte or SvelteKit project to your portfolio.",
                     "Mention Svelte stores, reactive declarations, and lifecycle.",
                     "Include Svelte in a side project and link the GitHub repo."],
    "bootstrap":    ["Add Bootstrap to your skills alongside CSS3 and HTML5.",
                     "Mention Bootstrap grid system and responsive utility classes.",
                     "Include a project built with Bootstrap for rapid UI development."],
    "tailwind css": ["Add Tailwind CSS to your skills section explicitly.",
                     "Mention utility-first design and custom Tailwind config.",
                     "Include a project using Tailwind with React or Vue."],
    "graphql":      ["Add GraphQL queries, mutations, and subscriptions to skills.",
                     "Mention Apollo Client or graphql-js in your tech stack.",
                     "Include a project with GraphQL API alongside REST APIs."],
    "redux":        ["Add Redux Toolkit to your skills and mention state management.",
                     "Mention Redux middleware (Thunk/Saga) and DevTools usage.",
                     "Include a React+Redux project with async data flow."],
    # Backend
    "node.js":      ["Add Node.js version and framework (Express/Fastify) explicitly.",
                     "Mention async I/O, event loop, and Node.js cluster usage.",
                     "Include a Node.js microservice or REST API project."],
    "express":      ["Add Express.js to your skills alongside Node.js.",
                     "Mention Express middleware, routing, and error handling.",
                     "Include Express REST API with authentication in a project."],
    "django":       ["Add Django and Django REST Framework to your skills.",
                     "Mention Django ORM, migrations, and admin panel usage.",
                     "Include a Django project with PostgreSQL backend."],
    "flask":        ["Add Flask to your skills with Python version.",
                     "Mention Flask Blueprints, SQLAlchemy, and Flask-RESTful.",
                     "Include a Flask REST API project with authentication."],
    "sql":          ["Add SQL explicitly alongside PostgreSQL/MySQL in skills.",
                     "Mention complex SQL queries, joins, indexing, and stored procs.",
                     "Include a project description mentioning SQL schema design."],
    "mongodb":      ["Add MongoDB and Mongoose to your skills section.",
                     "Mention aggregation pipelines, indexing, and schema design.",
                     "Include a MEAN/MERN stack project using MongoDB."],
    # DevOps
    "docker":       ["Add Docker and Docker Compose to your DevOps skills.",
                     "Mention Dockerfile creation and multi-stage builds.",
                     "Include containerized deployment in a project description."],
    "ci/cd":        ["Add CI/CD tools (GitHub Actions, Jenkins, GitLab CI) explicitly.",
                     "Mention automated testing, build, and deploy pipeline steps.",
                     "Describe a CI/CD pipeline you built in your experience section."],
    "git":          ["Add Git and mention branching strategy (Git Flow, trunk-based).",
                     "Mention pull request reviews, merge conflict resolution.",
                     "Include GitHub/GitLab profile link in your contact section."],
    "kubernetes":   ["Add Kubernetes to your DevOps skills section.",
                     "Mention pod management, deployments, services, and Helm.",
                     "Include a project deployed on Kubernetes cluster."],
    # ML/AI
    "python":       ["Add Python version (3.x) and key libraries to skills.",
                     "Mention Python in every relevant project description.",
                     "Include Python scripts or automation work in experience."],
    "pytorch":      ["Add PyTorch version and mention model training experience.",
                     "Include a PyTorch deep learning project with architecture details.",
                     "Mention custom training loops, loss functions, and optimizers."],
    "tensorflow":   ["Add TensorFlow/Keras to skills with version.",
                     "Include a TensorFlow project with model deployment details.",
                     "Mention TFX pipelines or TF Serving for production models."],
    "machine learning": ["Add specific ML algorithms to skills (XGBoost, Random Forest).",
                         "Include ML project with metrics (accuracy, F1, AUC-ROC).",
                         "Mention Scikit-Learn pipelines in your project descriptions."],
    "deep learning": ["Add deep learning architectures (CNN, RNN, Transformer) to skills.",
                      "Include a deep learning project with benchmark results.",
                      "Mention frameworks (PyTorch/TensorFlow) for deep learning."],
    "nlp":          ["Add NLP tasks (classification, NER, summarization) to skills.",
                     "Mention Hugging Face Transformers, SpaCy, or NLTK usage.",
                     "Include an NLP project with dataset and metric details."],
    "mlops":        ["Add MLOps tools (MLflow, Kubeflow, Airflow) to skills.",
                     "Mention model versioning, monitoring, and CI/CD for ML.",
                     "Include an end-to-end ML pipeline project description."],
    "scikit-learn": ["Add Scikit-Learn explicitly to your skills section.",
                     "Mention Scikit-Learn pipelines, cross-validation, and GridSearchCV.",
                     "Include a Scikit-Learn project with evaluation metrics."],
    "hugging face": ["Add Hugging Face Transformers to your NLP skills.",
                     "Mention fine-tuning BERT/GPT models for specific tasks.",
                     "Include a Hugging Face model hub project or contribution."],
    "automl":       ["Add AutoML tools (Optuna, H2O, AutoSklearn) to skills.",
                     "Mention hyperparameter tuning and automated model selection.",
                     "Include an AutoML experiment in a project description."],
    "computer vision": ["Add Computer Vision frameworks (OpenCV, YOLO) to skills.",
                        "Include a CV project with object detection or segmentation.",
                        "Mention dataset preparation and augmentation techniques."],
    "reinforcement learning": ["Add RL algorithms (PPO, DQN, A3C) to skills.",
                               "Include a Gym environment or RL project description.",
                               "Mention reward shaping and policy evaluation methods."],
    "data engineering": ["Add data pipeline tools (Airflow, Spark, Kafka) to skills.",
                         "Mention ETL pipeline design and data warehouse experience.",
                         "Include a data engineering project with scale metrics."],
    "feature engineering": ["Add feature engineering techniques to skills explicitly.",
                            "Mention encoding, scaling, embeddings, and PCA usage.",
                            "Include a project where feature engineering improved model."],
    # Cloud
    "cloud services": ["Add specific cloud platforms (AWS, Azure, GCP) to skills.",
                       "Mention cloud services used (EC2, S3, Lambda, Cloud Run).",
                       "Include a deployed cloud project with architecture details."],
    "aws":          ["Add specific AWS services (EC2, S3, Lambda, RDS) to skills.",
                     "Mention AWS certifications if you have them.",
                     "Include an AWS-deployed project with architecture details."],
    "responsive design": ["Add Responsive Design and mention CSS Grid/Flexbox.",
                          "Include mobile-first design approach in project descriptions.",
                          "Mention cross-browser testing and viewport meta usage."],
    "authentication & authorization": [
                     "Add Authentication & Authorization with JWT/OAuth2 to skills.",
                     "Mention Role-Based Access Control (RBAC) implementation.",
                     "Include a project with secure login and session management."],
    "restful apis": ["Add 'RESTful APIs' explicitly to your skills section.",
                     "Mention REST conventions (versioning, pagination, status codes).",
                     "Include a project where you designed and documented REST APIs."],
    "performance optimization": [
                     "Add performance metrics to project descriptions (load time, FPS).",
                     "Mention techniques: lazy loading, code splitting, caching.",
                     "Include Lighthouse or Web Vitals scores in project results."],
}

def get_suggestions_for_skill(skill):
    """Return rule-based suggestions for a missing skill."""
    key = skill.lower().strip()
    # Direct match
    if key in SKILL_SUGGESTIONS:
        return SKILL_SUGGESTIONS[key]
    # Partial match — find first key that is contained in the skill name
    for k, suggestions in SKILL_SUGGESTIONS.items():
        if k in key or key in k:
            return suggestions
    # Generic fallback
    return [
        f"Add '{skill}' explicitly to your Technical Skills section.",
        f"Include at least one project that demonstrates '{skill}' with measurable results.",
        f"Mention '{skill}' in your experience bullet points with specific context.",
    ]


class ResumeAnalysisAgent:
    def __init__(self, cutoff_score=75):
        self.cutoff_score = cutoff_score
        self.resume_text = None
        self.rag_vectorstore = None
        self.analysis_result = None
        self.jd_text = None
        self.extracted_skills = None
        self.resume_weakness = []
        self.resume_strengths = []
        self.improvement_suggestions = {}

    # ------------------------------------------------------------------
    # Text Extraction
    # ------------------------------------------------------------------

    def extract_text_from_pdf(self, pdf_file):
        try:
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
            if hasattr(pdf_file, 'getvalue'):
                data = pdf_file.getvalue()
            elif hasattr(pdf_file, 'read'):
                data = pdf_file.read()
            else:
                data = None

            reader = pypdf.PdfReader(io.BytesIO(data)) if data else pypdf.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            print(f"[DEBUG] PDF extracted: {len(text)} chars")
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_text(self, txt_file):
        try:
            if hasattr(txt_file, 'seek'):
                txt_file.seek(0)
            if hasattr(txt_file, 'getvalue'):
                raw_bytes = txt_file.getvalue()
            elif hasattr(txt_file, 'read'):
                raw_bytes = txt_file.read()
            else:
                with open(txt_file, 'rb') as f:
                    raw_bytes = f.read()

            for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
                try:
                    text = raw_bytes.decode(encoding)
                    print(f"[DEBUG] TXT extracted with {encoding}: {len(text)} chars")
                    return text
                except (UnicodeDecodeError, AttributeError):
                    continue

            return raw_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error extracting text from text file: {e}")
            return ""

    def extract_text_from_file(self, file):
        try:
            ext = file.name.split('.')[-1].lower() if hasattr(file, 'name') else str(file).split('.')[-1].lower()
            text = self.extract_text_from_pdf(file) if ext == 'pdf' else self.extract_text_from_text(file)
            if len(text) < 100:
                print(f"[WARNING] Extracted text very short: {len(text)} chars")
            return text
        except Exception as e:
            print(f"Error in extract_text_from_file: {e}")
            return ""

    # ------------------------------------------------------------------
    # Vector Store
    # ------------------------------------------------------------------

    def create_rag_vector_store(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        return FAISS.from_texts(chunks, get_embeddings())

    # ------------------------------------------------------------------
    # Skill Scoring
    # keyword_score (0-5) + semantic_score (0-5), capped at 10
    # Thresholds calibrated for all-MiniLM-L6-v2 (~0.25-0.55 range)
    # ------------------------------------------------------------------

    def score_skill(self, skill, resume_text):
        if not resume_text or len(resume_text.strip()) < 50:
            print(f"[WARNING] Empty resume_text in score_skill for '{skill}'")
            return 0, "Empty resume text"

        embeddings_model = get_embeddings()
        resume_lower = resume_text.lower()
        skill_lower = skill.lower()

        # Keyword frequency score
        exact_count = len(re.findall(r'\b' + re.escape(skill_lower) + r'\b', resume_lower))
        skill_words = [w for w in skill_lower.split() if len(w) > 3]
        word_count = sum(len(re.findall(r'\b' + re.escape(w) + r'\b', resume_lower)) for w in skill_words)
        total_mentions = exact_count + (word_count // 2)

        if total_mentions == 0:     keyword_score = 0
        elif total_mentions == 1:   keyword_score = 1
        elif total_mentions == 2:   keyword_score = 2
        elif total_mentions <= 5:   keyword_score = 3
        elif total_mentions <= 9:   keyword_score = 4
        else:                       keyword_score = 5

        # Semantic similarity score
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(resume_text)

        if not chunks:
            return min(keyword_score * 2, 10), f"Keyword only: {keyword_score * 2}/10"

        try:
            skill_vec = np.array(embeddings_model.embed_query(skill)).reshape(1, -1)
            chunk_vecs = np.array(embeddings_model.embed_documents(chunks))
            max_sim = float(np.max(cosine_similarity(skill_vec, chunk_vecs)))

            if max_sim < 0.15:    semantic_score = 0
            elif max_sim < 0.25:  semantic_score = 1
            elif max_sim < 0.32:  semantic_score = 2
            elif max_sim < 0.38:  semantic_score = 3
            elif max_sim < 0.44:  semantic_score = 4
            else:                 semantic_score = 5

        except Exception as e:
            print(f"[WARNING] Embedding failed for '{skill}': {e}")
            return min(keyword_score * 2, 10), f"Keyword only (embedding failed): {keyword_score * 2}/10"

        total_score = min(keyword_score + semantic_score, 10)
        reasoning = (
            f"Mentions: {total_mentions} → keyword {keyword_score}/5 | "
            f"Cosine: {max_sim:.3f} → semantic {semantic_score}/5 | "
            f"Total: {total_score}/10"
        )
        return total_score, reasoning

    # ------------------------------------------------------------------
    # Semantic Skill Analysis
    # ------------------------------------------------------------------

    def semantic_skill_analysis(self, resume_text, skills):
        print(f"[DEBUG] resume_text length: {len(resume_text)}")
        skill_scores = {}
        skill_reasoning = {}
        missing_skills = []
        total_score = 0

        for skill in skills:
            score, reasoning = self.score_skill(skill, resume_text)
            skill_scores[skill] = score
            skill_reasoning[skill] = reasoning
            total_score += score
            if score <= 5:
                missing_skills.append(skill)
            print(f"[DEBUG] {skill}: {score}/10")

        overall_score = int((total_score / (10 * len(skills))) * 100)
        selected = overall_score >= self.cutoff_score
        strengths = [s for s, sc in skill_scores.items() if sc >= 7]
        improvement_areas = missing_skills if not selected else []
        self.resume_strengths = strengths

        print(f"[DEBUG] Overall: {overall_score}")
        return {
            "overall_score": overall_score,
            "skill_scores": skill_scores,
            "skill_reasoning": skill_reasoning,
            "selected": selected,
            "reasoning": "Scored using keyword frequency + semantic similarity.",
            "missing_skills": missing_skills,
            "strengths": strengths,
            "improvement_areas": improvement_areas
        }

    # ------------------------------------------------------------------
    # Weakness Analysis — FIXED: rule-based instead of broken LLM
    #
    # flan-t5-base produces nonsense suggestions like:
    #   "Make sure resume is not an HTML resume"
    #   "Make sure you have a picture of the Russian flag"
    # This is because flan-t5 cannot follow instruction prompts reliably
    # for open-ended generation tasks.
    # Solution: use the SKILL_SUGGESTIONS dict above — always accurate.
    # ------------------------------------------------------------------

    def analyze_resume_weakness(self):
        if not self.resume_text or not self.extracted_skills or not self.analysis_result:
            return []

        weaknesses = []

        for skill in self.analysis_result.get("missing_skills", []):
            suggestions = get_suggestions_for_skill(skill)

            weakness_detail = {
                "skill": skill,
                "score": self.analysis_result.get("skill_scores", {}).get(skill, 0),
                "detail": f"Limited or no evidence of '{skill}' found in the resume.",
                "suggestions": suggestions,
                "example": f"Add '{skill}' explicitly in your Technical Skills section "
                           f"and reference it in at least 2 project or experience bullet points."
            }
            weaknesses.append(weakness_detail)
            self.improvement_suggestions[skill] = {
                "suggestions": suggestions,
                "example": weakness_detail["example"]
            }

        self.resume_weakness = weaknesses
        return weaknesses

    # ------------------------------------------------------------------
    # JD Skill Extraction
    # ------------------------------------------------------------------

    def extract_skills_from_jd(self, jd_text):
        try:
            llm = get_llm()
            prompt = (
                f"List the technical skills required in this job description "
                f"as a comma-separated list. No explanations.\n\n{jd_text[:1500]}"
            )
            response = llm(prompt, max_new_tokens=150)[0]['generated_text']
            skills_text = response.replace(prompt, "").strip()

            match = re.search(r'\[(.*?)\]', skills_text, re.DOTALL)
            if match:
                try:
                    skills_list = ast.literal_eval(match.group(0))
                    if isinstance(skills_list, list):
                        return [s.strip() for s in skills_list if s.strip()]
                except Exception:
                    pass

            raw = re.split(r'[,\n]', skills_text)
            return [s.strip().strip('-*•').strip() for s in raw if s.strip()]

        except Exception as e:
            print(f"Error extracting skills from JD: {e}")
            return []

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def analyze_resume(self, resume_file, role_requirements=None, custom_jd=None):
        self.resume_text = self.extract_text_from_file(resume_file)

        if not self.resume_text or len(self.resume_text.strip()) < 50:
            print("[ERROR] Resume text extraction failed.")
            return {
                "overall_score": 0, "skill_scores": {}, "skill_reasoning": {},
                "selected": False, "reasoning": "Could not extract text from file.",
                "missing_skills": [], "strengths": [], "improvement_areas": [],
                "error": "Failed to read resume. Please re-upload and try again."
            }

        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp:
            tmp.write(self.resume_text)
            self.resume_file_path = tmp.name

        self.rag_vectorstore = self.create_rag_vector_store(self.resume_text)

        if custom_jd:
            self.jd_text = self.extract_text_from_file(custom_jd)
            self.extracted_skills = self.extract_skills_from_jd(self.jd_text)
            self.analysis_result = self.semantic_skill_analysis(self.resume_text, self.extracted_skills)
        elif role_requirements:
            self.extracted_skills = role_requirements
            self.analysis_result = self.semantic_skill_analysis(self.resume_text, role_requirements)

        if self.analysis_result and self.analysis_result.get("missing_skills"):
            self.analyze_resume_weakness()
            self.analysis_result["detailed_weaknesses"] = self.resume_weakness

        return self.analysis_result

    # ------------------------------------------------------------------
    # Resume Q&A
    # ------------------------------------------------------------------

    def ask_question(self, question):
        if not self.rag_vectorstore or not self.resume_text:
            return "Please analyze a resume first."

        retriever = self.rag_vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = get_llm()
        docs = retriever.invoke(question)
        context = " ".join([doc.page_content for doc in docs])[:2000]
        prompt = f"Answer based on the resume.\n\nResume:\n{context}\n\nQuestion: {question}\nAnswer:"
        result = llm(prompt, max_new_tokens=200)[0]['generated_text']
        answer = result.replace(prompt, "").strip()
        return answer if answer else "Could not find relevant information in the resume."

    # ------------------------------------------------------------------
    # Interview Questions
    # ------------------------------------------------------------------

    # def generate_interview_questions(self, question_types, difficulty_level, num_questions):
    #     if not self.resume_text or not self.extracted_skills:
    #         return []
    #     try:
    #         llm = get_llm()
    #         skills_str = ', '.join(self.extracted_skills[:8])
    #         prompt = (
    #             f"Generate {num_questions} {difficulty_level} interview questions "
    #             f"for a candidate skilled in: {skills_str}. "
    #             f"Question types: {', '.join(question_types)}. "
    #             f"Format each as: Type: Question"
    #         )
    #         response = llm(prompt, max_new_tokens=300)[0]['generated_text']
    #         output = response.replace(prompt, "").strip()

    #         questions = []
    #         for line in output.split('\n'):
    #             line = line.strip()
    #             if ':' in line:
    #                 parts = line.split(':', 1)
    #                 q_type, q_text = parts[0].strip(), parts[1].strip()
    #                 if q_text:
    #                     matched = next(
    #                         (t for t in question_types if t.lower() in q_type.lower()),
    #                         question_types[0] if question_types else "General"
    #                     )
    #                     questions.append((matched, q_text))

    #         if not questions and output:
    #             questions = [(question_types[0] if question_types else "General", output)]
    #         return questions[:num_questions]
    #     except Exception as e:
    #         print(f"Error generating interview questions: {e}")
    #         return []

    def generate_interview_questions(self, question_types, difficulty_level, num_questions):
        if not self.resume_text or not self.extracted_skills:
            return []

        try:
            llm = get_llm()

            # -------------------------------
            # ✅ Use RAG for better context
            # -------------------------------
            try:
                retriever = self.rag_vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke("skills projects experience")
                resume_context = " ".join([d.page_content for d in docs])[:1000]
            except:
                resume_context = self.resume_text[:1000]

            skills_str = ', '.join(self.extracted_skills[:6])

            questions = []
            used_questions = set()

            # -------------------------------
            # 🔁 Retry mechanism (important)
            # -------------------------------
            max_attempts = num_questions * 3
            attempt = 0

            while len(questions) < num_questions and attempt < max_attempts:
                attempt += 1

                prompt = f"""
                Generate ONE interview question STRICTLY based on the resume.

                Resume:
                {resume_context}

                Skills:
                {skills_str}

                STRICT RULES:
                - Question MUST reference a skill, tool, or project from the resume
                - Do NOT ask generic HR questions
                - Do NOT invent information
                - If resume mentions Python → ask about Python
                - If resume mentions project → ask about project
                - Question MUST end with '?'

                BAD EXAMPLES:
                - How would you describe yourself?
                - What are your strengths?
                - What is the minimum qualification?

                GOOD EXAMPLES:
                - How did you use Python in your project?
                - Explain your experience with machine learning models?
                - How did you implement REST APIs in your application?

                ONLY output the question.
                """

                response = llm(prompt, max_new_tokens=60)[0]['generated_text']
                output = response.replace(prompt, "").strip()

                print(f"[DEBUG] Attempt {attempt}:", output)

                # -------------------------------
                # ✅ Extract valid question
                # -------------------------------
                if '?' in output:
                    q = output.split('?')[0].strip() + '?'

                    # remove very short or bad questions
                    if len(q) > 20 and q.lower() not in used_questions:
                        questions.append(q)
                        used_questions.add(q.lower())

            # -------------------------------
            # 🚫 Remove generic questions
            # -------------------------------
            bad_patterns = [
                "most important",
                "what is important",
                "tell me about",
                "describe yourself",
                "strength",
                "weakness",
                "minimum qualification",
                "who are you"
            ]

            filtered_questions = []
            for q in questions:
                if not any(bp in q.lower() for bp in bad_patterns):
                    filtered_questions.append(q)

            if filtered_questions:
                questions = filtered_questions

            # -------------------------------
            # 🚨 Final fallback (if still empty)
            # -------------------------------
            if not questions:
                print("[ERROR] No valid questions generated, using fallback")

                questions = [
                    "Explain a project from your resume and the challenges you faced?",
                    "How did you use your main skill in your project?",
                    "What technologies did you use in your recent work?",
                    "How would you improve performance in your application?"
                ]

            # -------------------------------
            # 🎯 Assign question types
            # -------------------------------
            final_questions = []

            for i, q in enumerate(questions[:num_questions]):
                if question_types:
                    q_type = question_types[i % len(question_types)]
                else:
                    q_type = "General"

                final_questions.append((q_type, q))

            print("[DEBUG] Final Questions:", final_questions)

            return final_questions

        except Exception as e:
            print(f"Error generating interview questions: {e}")
            return []

    # ------------------------------------------------------------------
    # Resume Improvement
    # ------------------------------------------------------------------

    def improve_resume(self, improvement_areas, target_role=""):
        if not self.resume_text:
            return {}
        try:
            improvements = {}
            llm = get_llm()

            for area in improvement_areas:
                if area == "Skills Highlighting" and self.resume_weakness:
                    specific = []
                    before_after = {"before": "", "after": ""}
                    for weakness in self.resume_weakness:
                        skill_name = weakness.get("skill", "")
                        for s in weakness.get("suggestions", []):
                            specific.append(f"**{skill_name}**: {s}")
                        if weakness.get("example"):
                            lines = self.resume_text.split('\n')
                        for i, line in enumerate(lines):
                            if skill_name.lower() in line.lower():
                                # Grab just that line and 2 lines around it
                                context = '\n'.join(lines[max(0, i-1):i+3])
                                before_after = {
                                    "before": context.strip(),
                                    "after": context.strip() + "\n• " + weakness["example"]
                                }
                                break
                    improvements["Skills Highlighting"] = {
                        "description": "Highlight skills more explicitly in your resume.",
                        "specific": specific,
                        "before_after": before_after
                    }

            remaining = [a for a in improvement_areas if a not in improvements]
            for area in remaining:
                missing_str = ', '.join(self.analysis_result.get('missing_skills', [])[:5])
                # Use rule-based suggestions for missing skills instead of LLM
                area_suggestions = []
                for skill in self.analysis_result.get('missing_skills', [])[:5]:
                    area_suggestions.extend(get_suggestions_for_skill(skill)[:1])

                improvements[area] = {
                    "description": f"Your resume needs improvement in {area}. "
                                   f"Missing skills: {missing_str}.",
                    "specific": area_suggestions[:5] if area_suggestions
                               else ["Review and enhance this section."],
                    "before_after": {"before": "", "after": ""}
                }

            # for area in improvement_areas:
            #     if area not in improvements:
            #         improvements[area] = {
            #             "description": f"Improvement needed in {area}.",
            #             "specific": ["Review and enhance this section."],
            #             "before_after": {"before": "", "after": ""}
            #         }
            return improvements

        except Exception as e:
            print(f"Error generating resume improvements: {e}")
            return {
                area: {"description": "Error", "specific": [], "before_after": {"before": "", "after": ""}}
                for area in improvement_areas
            }

    # ------------------------------------------------------------------
    # Improved Resume Generator
    # ------------------------------------------------------------------

    def get_improved_resume(self, target_role="", highlight_skills=""):
        if not self.resume_text:
            return "Please upload and analyze a resume first."
        try:
            llm = get_llm()
            skills_to_highlight = []

            if highlight_skills:
                skills_to_highlight = [s.strip() for s in highlight_skills.split(",") if s.strip()]

            if not skills_to_highlight and self.analysis_result:
                skills_to_highlight = list(self.analysis_result.get('missing_skills', []))
                skills_to_highlight += [s for s in self.analysis_result.get('strengths', [])
                                        if s not in skills_to_highlight]
                if self.extracted_skills:
                    skills_to_highlight += [s for s in self.extracted_skills
                                            if s not in skills_to_highlight]

            weakness_notes = ""
            if self.resume_weakness:
                weakness_notes = "Weaknesses to address:\n"
                for w in self.resume_weakness:
                    weakness_notes += f"- {w.get('skill','')}: {w.get('detail','')}\n"

            jd_context = ""
            if self.jd_text:
                jd_context = f"Job Description:\n{self.jd_text[:500]}\n\n"
            elif target_role:
                jd_context = f"Target Role: {target_role}\n\n"

            prompt = (
                f"Rewrite this resume to highlight: {', '.join(skills_to_highlight[:10])}. "
                f"{jd_context}{weakness_notes}"
                f"Add quantifiable achievements and ATS-friendly formatting.\n\n"
                f"Resume:\n{self.resume_text[:2000]}"
            )
            response = llm(prompt, max_new_tokens=512)[0]['generated_text']
            improved_resume = response.replace(prompt, "").strip() or self.resume_text

            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmp:
                tmp.write(improved_resume)
                self.improved_resume_path = tmp.name

            return improved_resume
        except Exception as e:
            print(f"Error generating improved resume: {e}")
            return "Error generating improved resume. Please try again."

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        try:
            if hasattr(self, 'resume_file_path') and os.path.exists(self.resume_file_path):
                os.unlink(self.resume_file_path)
            if hasattr(self, 'improved_resume_path') and os.path.exists(self.improved_resume_path):
                os.unlink(self.improved_resume_path)
        except Exception as e:
            print(f"Error cleaning up: {e}")


if __name__ == "__main__":
    print("Agent file executed successfully")