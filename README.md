# SymptomsChecker

## Project Description

SymptomsChecker is a machine learning-based application designed to help users identify potential health conditions based on their symptoms. It leverages a trained deep learning model to provide accurate and timely suggestions.

## Installation

### Data Handling & Preprocessing

```bash
pip install pandas numpy scikit-learn
```

### Deep Learning (Model Training)

```bash
pip install tensorflow keras
```

### API Backend (FastAPI)

```bash
pip install fastapi uvicorn
```

### Model Deployment & Communication

```bash
pip install pydantic python-multipart
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/mozaddedalfeshani/symptoms-checker.git
   cd symptoms-checker
   ```

2. Navigate to the backend directory and start the FastAPI server:

   ```bash
   cd backend
   uvicorn main:app --reload
   ```

3. Navigate to the frontend directory and start the React application:

   ```bash
   cd frontend
   npm install
   npm start
   ```

4. Access the application at `http://localhost:3000`.

## Project Structure

```
symptoms-checker/
│── backend/ # FastAPI backend
│ ├── main.py # FastAPI app
│ ├── model.py # Deep Learning model
│ ├── requirements.txt
│── frontend/ # React.js frontend
│ ├── src/
│ ├── App.tsx # React App
│── dataset/ # Store your dataset here
│── models/ # Trained ML model
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License.
