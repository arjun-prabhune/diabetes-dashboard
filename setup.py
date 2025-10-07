"""
Automated setup script for Diabetes Risk Dashboard
Handles directory creation, dependency checking, and initial setup
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60 + "\n")

def print_step(step_num, description):
    """Print a step indicator"""
    print(f"[Step {step_num}] {description}")

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print_step(1, "Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. You have Python {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_project_structure():
    """Create necessary directories"""
    print_step(2, "Creating project structure...")
    
    directories = ['data', 'models', 'utils']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created/verified: {directory}/")
    
    return True

def check_pip():
    """Check if pip is installed"""
    print_step(3, "Checking pip...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                      check=True, capture_output=True)
        print("  ✓ pip is installed")
        return True
    except subprocess.CalledProcessError:
        print("  ❌ pip is not installed")
        return False

def install_dependencies():
    """Install required Python packages"""
    print_step(4, "Installing dependencies...")
    
    packages = [
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'joblib>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'plotly>=5.14.0',
        'streamlit>=1.28.0',
        'shap>=0.42.0'
    ]
    
    print("  Installing packages (this may take a few minutes)...")
    
    try:
        for package in packages:
            package_name = package.split('>=')[0]
            print(f"    Installing {package_name}...", end=' ')
            
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package, '-q'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✓")
            else:
                print(f"⚠ (may already be installed)")
        
        print("\n  ✓ All dependencies installed successfully")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Error installing dependencies: {e}")
        return False

def verify_installation():
    """Verify that all required packages are importable"""
    print_step(5, "Verifying installation...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'joblib': 'joblib',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
        'shap': 'shap'
    }
    
    all_installed = True
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - Failed to import")
            all_installed = False
    
    return all_installed

def check_required_files():
    """Check if all required Python files exist"""
    print_step(6, "Checking required files...")
    
    required_files = {
        'app.py': 'Streamlit application',
        'train_model.py': 'Model training script',
        'utils/preprocessing.py': 'Preprocessing utilities',
        'config.py': 'Configuration file (optional)',
        'requirements.txt': 'Dependencies list'
    }
    
    all_exist = True
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            print(f"  ✓ {file_path} - {description}")
        else:
            if 'optional' in description.lower():
                print(f"  ⚠ {file_path} - {description} (not found, but optional)")
            else:
                print(f"  ❌ {file_path} - {description} (REQUIRED)")
                all_exist = False
    
    return all_exist

def print_next_steps():
    """Print instructions for next steps"""
    print_header("SETUP COMPLETE!")
    
    print("🎉 Your Diabetes Risk Dashboard is ready to use!\n")
    
    print("NEXT STEPS:")
    print("-" * 60)
    print("\n1. Train the machine learning model:")
    print("   → python train_model.py\n")
    print("2. Launch the dashboard:")
    print("   → streamlit run app.py\n")
    print("3. Open your browser to:")
    print("   → http://localhost:8501\n")
    print("-" * 60)
    
    print("\nQUICK TIPS:")
    print("  • First time? The training step takes 1-2 minutes")
    print("  • The dashboard will auto-open in your browser")
    print("  • All your data stays local - nothing is sent online")
    print("  • Check README.md for detailed documentation")
    
    print("\nTROUBLESHOOTING:")
    print("  • Port 8501 in use? → streamlit run app.py --server.port 8502")
    print("  • Missing SHAP plots? → pip install shap")
    print("  • Need help? → Check QUICKSTART.md")
    
    print("\n" + "="*60)

def run_setup():
    """Main setup function"""
    print_header("DIABETES RISK DASHBOARD - SETUP")
    
    print("This script will:")
    print("  1. Check Python version (3.8+ required)")
    print("  2. Create project directories")
    print("  3. Verify pip installation")
    print("  4. Install Python dependencies")
    print("  5. Verify installations")
    print("  6. Check required files")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Please upgrade Python to 3.8 or higher")
        return False
    
    # Step 2: Create directories
    if not create_project_structure():
        print("\n❌ Setup failed: Could not create directories")
        return False
    
    # Step 3: Check pip
    if not check_pip():
        print("\n❌ Setup failed: pip is required")
        print("   Install pip: https://pip.pypa.io/en/stable/installation/")
        return False
    
    # Step 4: Install dependencies
    print("\n⚠️  Installing dependencies may take 2-5 minutes...")
    user_input = input("Continue with installation? (y/n): ")
    
    if user_input.lower() != 'y':
        print("\n⚠️  Setup cancelled. Run this script again when ready.")
        return False
    
    if not install_dependencies():
        print("\n⚠️  Some dependencies may not have installed correctly")
        print("   Try manually: pip install -r requirements.txt")
    
    # Step 5: Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages could not be verified")
        print("   The dashboard may still work - try running it")
    
    # Step 6: Check files
    if not check_required_files():
        print("\n⚠️  Some required files are missing")
        print("   Make sure all Python files are in the correct locations")
    
    # Print next steps
    print_next_steps()
    
    return True

def main():
    """Entry point"""
    try:
        success = run_setup()
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()