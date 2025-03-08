# CIFAR-10 Project

## Description
This repository contains an implementation of the CIFAR-10 classification project. It includes data processing, model training, and evaluation scripts.

## Project Structure
```
CIFAR_10/
│── data/        # Dataset files (ignored in Git)
│── src/         # Source code for training and evaluation
│── venv/        # Virtual environment (ignored in Git)
│── .env         # Environment variables (ignored in Git)
│── .gitignore   # Git ignore file
│── README.md    # Project documentation
```

## Setup Instructions

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/mithunveluru/CIFAR_10.git
   cd CIFAR_10
   ```

2. **Create and Activate Virtual Environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate     # For Windows
   ```

3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Project:**
   ```sh
   python src/train.py
   ```

## Git Ignore Setup
This repository includes a `.gitignore` file to exclude unnecessary files such as:

```
.env
venv/
__pycache__/
data/
```
To apply `.gitignore` to files that were already tracked, use:
```sh
git rm -r --cached .env venv/
git commit -m "Removed ignored files from tracking"
git push origin main
```

## Contributing
Feel free to fork the repository and create a pull request with improvements!

## License
This project is licensed under the MIT License.

