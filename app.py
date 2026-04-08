import streamlit as st

# This must be the first Streamlit command
st.set_page_config(
    page_title="SS Recruiter Agent",
    page_icon= "🚀",
    layout = "wide"
)

import ui
from agent import ResumeAnalysisAgent
import atexit



ROLE_Requirements = {
    "AI/ML Engineer": [
        # Programming
        "Python", "R", "Java", "C++",

        # Core ML/DL
        "Machine Learning", "Deep Learning", "Supervised Learning",
        "Unsupervised Learning", "Reinforcement Learning",

        # Frameworks
        "TensorFlow", "Keras", "PyTorch", "Scikit-learn", "XGBoost", "LightGBM",

        # NLP
        "NLP", "Natural Language Processing", "Transformers",
        "Hugging Face", "BERT", "GPT", "LLMs", "Text Classification",

        # Computer Vision
        "Computer Vision", "OpenCV", "Image Processing",
        "Object Detection", "YOLO", "CNN",

        # Data Handling
        "Pandas", "NumPy", "Data Cleaning", "Data Preprocessing",
        "Feature Engineering", "Feature Selection",

        # MLOps
        "MLOps", "Model Deployment", "Model Monitoring",
        "MLflow", "Kubeflow", "Docker", "CI/CD",

        # Data Engineering overlap
        "Data Engineering", "ETL", "Big Data", "Spark",

        # AutoML
        "AutoML", "Hyperparameter Tuning", "Grid Search", "Random Search"
    ],

    "Frontend Engineer": [
        # Core
        "HTML", "HTML5", "CSS", "CSS3", "JavaScript", "TypeScript",

        # Frameworks
        "React", "Vue", "Angular", "Next.js", "Nuxt.js", "Svelte",

        # Styling
        "Bootstrap", "Tailwind CSS", "Material UI", "SASS", "SCSS",

        # State Management
        "Redux", "Zustand", "Context API",

        # APIs
        "REST APIs", "GraphQL", "Axios", "Fetch API",

        # Performance
        "Performance Optimization", "Lazy Loading", "Code Splitting",

        # Advanced
        "WebAssembly", "Three.js", "D3.js",

        # Testing
        "Jest", "Cypress", "Testing Library"
    ],

    "Backend Engineer": [
        # Languages
        "Python", "Java", "Node.js", "Go", "C#", "Ruby",

        # Frameworks
        "Django", "Flask", "FastAPI", "Spring Boot", "Express.js",

        # APIs
        "REST APIs", "GraphQL", "gRPC", "Microservices",

        # Databases
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "NoSQL",
        "Redis", "Cassandra", "DynamoDB",

        # Messaging
        "Kafka", "RabbitMQ", "Message Queues",

        # DevOps overlap
        "Docker", "Kubernetes", "CI/CD",

        # Cloud
        "AWS", "Azure", "GCP",

        # Concepts
        "Authentication", "Authorization", "OAuth", "JWT",
        "Scalability", "System Design"
    ],

    "Data Engineer": [
        # Core
        "Python", "SQL", "Scala", "Java",

        # Big Data
        "Apache Spark", "Hadoop", "Hive",

        # Streaming
        "Kafka", "Flink",

        # ETL
        "ETL Pipelines", "Data Pipelines", "Data Integration",

        # Orchestration
        "Airflow", "Prefect",

        # Warehousing
        "Data Warehousing", "Snowflake", "BigQuery", "Redshift",

        # Cloud
        "AWS Glue", "Azure Data Factory", "GCP Dataflow",

        # Modeling
        "DBT", "Data Modeling", "Star Schema", "Snowflake Schema"
    ],

    "DevOps Engineer": [
        # Containers
        "Docker", "Kubernetes", "Helm",

        # IaC
        "Terraform", "CloudFormation",

        # CI/CD
        "CI/CD", "Jenkins", "GitHub Actions", "GitLab CI",

        # Cloud
        "AWS", "Azure", "GCP",

        # Monitoring
        "Prometheus", "Grafana", "ELK Stack", "Datadog",

        # Config Mgmt
        "Ansible", "Chef", "Puppet",

        # OS & Networking
        "Linux", "Shell Scripting", "Networking",

        # Concepts
        "Site Reliability Engineering", "High Availability",
        "Auto Scaling", "Load Balancing"
    ],

    "Full Stack Developer": [
        # Frontend
        "HTML", "CSS", "JavaScript", "TypeScript",
        "React", "Angular", "Vue",

        # Backend
        "Node.js", "Express.js", "Django", "Flask",

        # Databases
        "MongoDB", "MySQL", "PostgreSQL",

        # APIs
        "REST APIs", "GraphQL",

        # DevOps
        "Docker", "CI/CD", "Cloud Services",

        # Concepts
        "Authentication", "Authorization", "JWT",
        "Responsive Design", "System Design",

        # Tools
        "Git", "GitHub", "Bitbucket"
    ],

    "Product Manager": [
        "Product Strategy", "Product Roadmap", "Roadmapping",
        "User Research", "User Interviews",

        "Agile", "Scrum", "Kanban",

        "Market Research", "Competitive Analysis",

        "Stakeholder Management", "Cross-functional Teams",

        "Data Analysis", "SQL", "Excel",

        "A/B Testing", "Experimentation",

        "KPI Definition", "OKRs", "Metrics",

        "User Stories", "PRD", "Product Lifecycle",

        "Customer Journey Mapping", "UX Thinking"
    ],

    "Data Scientist": [
        # Languages
        "Python", "R", "SQL",

        # Core
        "Machine Learning", "Statistics",
        "Probability", "Hypothesis Testing",

        # Libraries
        "Pandas", "NumPy", "Scikit-learn",
        "Matplotlib", "Seaborn",

        # Advanced
        "Deep Learning", "NLP",

        # Concepts
        "Feature Engineering", "Model Evaluation",
        "Cross Validation",

        # Tools
        "Jupyter Notebook", "Google Colab",

        # Visualization
        "Data Visualization", "Tableau", "Power BI"
    ]
}

# Initialize session state variables
if 'resume_agent' not in st.session_state:
    st.session_state.resume_agent = None

if 'resume_analyzed' not in st.session_state:
    st.session_state.resume_analyzed = False

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None


# Important part to check
def setup_agent(config):
    """
    Set up resume analysis agent with provided configuration
    """
    # if not config["openai_api_key"]:
    #     st.error("⚠️ Please enter your Open API Key in the sidebar")
    #     return None
    
    # Initialize or update the agent with the API key
    # if st.session_state.resume_agent is None:
    #     st.session_state.resume_agent = ResumeAnalysisAgent(api_key=config["openai_api_key"])

    # else:
    #     st.session_state.resume_agent.api_key = config["openai_api_key"]
    
    if st.session_state.get("resume_agent") is None:
        st.session_state.resume_agent = ResumeAnalysisAgent()


    return st.session_state.resume_agent

def analyze_resume(agent, resume_file,role, custom_jd):
    """Analyze the resume with the agent"""
    if not resume_file:
        st.error("⚠️ Please upload a resume.")
        return None
    
    try:
        with st.spinner("🔍 Analysing resume... This may take a minute."):
            if custom_jd:
                result = agent.analyze_resume(resume_file,custom_jd=custom_jd)
            else:
                result = agent.analyze_resume(resume_file,role_requirements = ROLE_Requirements[role])

            
            st.session_state.resume_analyzed = True
            st.session_state.analysis_result = result
            return result
    except Exception as e:
        st.error(f"⚠️Error analyzing resume:{e}")
        return None
    
def ask_question(agent,question):
    """Ask a question about the resume"""
    try:
        with st.spinner("Generating response..."):
            response = agent.ask_question(question)
            return response
    except Exception as e:
        return f"Error:{e}"
    
def generate_interview_questions(agent,question_type,difficulty,num_questions):
    """Generate interview questions based on the resume"""
    try:
        with st.spinner("Generating personalized interview questions..."):
            questions = agent.generate_interview_questions(question_type,difficulty,num_questions)
            return questions
    except Exception as e:
        st.error(f"⚠️ Error generating questions:{e}")
        return []

def improve_resume(agent, improvement_areas, target_role):
    """Generate resume improvement suggestions"""
    try:
        return agent.improve_resume(improvement_areas, target_role)  # ← removed st.spinner wrapper
    except Exception as e:
        st.error(f"⚠️ Error generating improvements: {e}")
        return {}
    
def get_improved_resume(agent, target_role,highligth_skills):
    """Get an improved version of the resume"""
    try:
        with st.spinner("Creating improved resume..."):
            return agent.get_improved_resume(target_role,highligth_skills)
        
    except Exception as e:
        st.error(f"⚠️ Error creating improved resume")
        return "Error generating improved resume"

def cleanup():
    """Clean up resources when the app exits"""
    if "resume_agent" in st.session_state and st.session_state.resume_agent:
        st.session_state.resume_agent = None
    # if st.session_state.resume_agent:
    #     st.session_state.resume_agent.cleanup()

# register cleanup function
atexit.register(cleanup)

def main():
    #Setup page UI
    ui.setup_page()
    ui.display_header()

    # Set up sidebar and get configuration
    config = ui.setup_sidebar()

    # Set up the agent
    agent = setup_agent(config)

    # Create tabs for different functionalities
    tabs = ui.create_tabs()

    # Tab 1: Resume Analysis
    with tabs[0]:
        role, custom_jd = ui.role_selection_section(ROLE_Requirements)
        uploaded_resume = ui.resume_upload_section()

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("🔍Analyze Resume",type="primary"):
                if agent and uploaded_resume:
                    # just store the result, don't display it here
                    analyze_resume(agent,uploaded_resume,role,custom_jd)
        
        # Display analysis result (only once)
        if st.session_state.analysis_result:
            ui.display_analysis_results(st.session_state.analysis_result)

    # Tab 2 : Resume Q&A
    with tabs[1]:
        # We need to ensure the agent and resume are available
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_qa_section(
                has_resume=True, # Explicitly set to True since we checked above
                ask_question_func=lambda q: ask_question(st.session_state.resume_agent,q)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    # Tab 3: Interview Questions
    with tabs[2]:
        # We need to ensure the agent and resume are available
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.interview_questions_section(
                has_resume=True, # Explicitly set to True since we check
                generate_question_func=lambda types, diff, num:
                generate_interview_questions(st.session_state.resume_agent,types,diff,num)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    
    # Tab 4: Resume Improvement
    with tabs[3]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_improvement_section(
                has_resume=True,
                improve_resume_func=lambda areas,role:improve_resume(st.session_state.resume_agent,areas,role)
            )
        else:
            st.warning("Please upload and analyze a resume first in 'Resume Analysis' tab.")

    # Tab 5: Improved Resume
    with tabs[4]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.improved_resume_section(
                has_resume=True,
                get_improved_resume_func=lambda role, skills: get_improved_resume
                (st.session_state.resume_agent,role, skills)
            )
        else:
            st.warning("Please upload and analyze a resume first in 'Resume Analysis' tab.")

    
if __name__ == "__main__":
    main()
    