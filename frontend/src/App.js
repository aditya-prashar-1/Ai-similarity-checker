import React, { useState } from 'react';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [code, setCode] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [referenceCode, setReferenceCode] = useState('');

  const sampleQuestions = [
    "Write a function to reverse a number",
    "Implement binary search in Python",
    "Create a function to check if a number is prime",
    "Write a program to find the factorial of a number",
    "Implement a stack data structure in Python"
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question || !code) {
      setError('Both question and code are required');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setReferenceCode('');

    try {
      const response = await fetch('http://localhost:5000/generate_and_compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, code })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'API request failed');
      }

      const data = await response.json();
      setResult(data);
      setReferenceCode(data.reference_code);
    } catch (err) {
      setError(err.message || 'Error processing request');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleQuestionSelect = (q) => {
    setQuestion(q);
  };

  return (
      <div className="app">
        <header className="header">
          <h1>Code Plagiarism Detector</h1>
          <p>AI-powered code similarity analysis</p>
        </header>

        <div className="container">
          <div className="question-bank">
            <h3>Sample Questions</h3>
            <ul>
              {sampleQuestions.map((q, index) => (
                  <li
                      key={index}
                      onClick={() => handleQuestionSelect(q)}
                      className={question === q ? 'active' : ''}
                  >
                    {q}
                  </li>
              ))}
            </ul>
          </div>

          <form onSubmit={handleSubmit} className="code-form">
            <div className="form-group">
              <h3>Programming Question</h3>
              <textarea
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Enter programming question..."
                  rows={3}
              />
            </div>

            <div className="code-inputs">
              <div className="code-section">
                <h3>Your Solution</h3>
                <textarea
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    placeholder="Enter your code solution..."
                    rows={15}
                />
              </div>

              <div className="code-section">
                <h3>AI-Generated Reference</h3>
                <div className="reference-code">
                  {referenceCode ? (
                      <pre>{referenceCode}</pre>
                  ) : (
                      <p>Reference code will appear here after analysis</p>
                  )}
                </div>
              </div>
            </div>

            <button
                type="submit"
                disabled={loading || !question || !code}
                className="submit-btn"
            >
              {loading ? 'Analyzing...' : 'Generate & Compare'}
            </button>
          </form>

          {error && <div className="error">{error}</div>}

          {result && (
              <div className={`result ${result.prediction}`}>
                <h2>Analysis Result</h2>
                <div className="score-container">
                  <div className="score-circle">
                    <span>{result.confidence}</span>
                    <div className="circle-bg">
                      <div
                          className="circle-progress"
                          style={{
                            background: `conic-gradient(var(--success) 0%, transparent ${result.similarity_score * 180}deg)`,
                            transform: `rotate(${result.similarity_score * 180}deg)`
                          }}
                      ></div>
                    </div>
                  </div>
                  <div className="score-details">
                    <p>Similarity Score: <strong>{result.similarity_score.toFixed(4)}</strong></p>
                    <p>Prediction:
                      <strong className={result.prediction}>
                        {result.prediction}
                      </strong>
                    </p>
                  </div>
                </div>

                <div className="interpretation">
                  {result.similarity_score >= 0.7 ? (
                      <p>High similarity to reference implementation. Your solution closely matches the AI-generated code.</p>
                  ) : result.similarity_score >= 0.5 ? (
                      <p>Moderate similarity. Your solution shares some logic with the reference implementation.</p>
                  ) : (
                      <p>Low similarity. Your solution appears to be significantly different from the reference.</p>
                  )}
                </div>
              </div>
          )}
        </div>

        <footer>
          <p>Powered by CodeBERT and CodeGen | AI Code Analysis</p>
        </footer>
      </div>
  );
}

export default App;