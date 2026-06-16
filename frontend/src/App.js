import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [pdfName, setPdfName] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const uploadPdf = async () => {
    if (!file) return;
    setUploadStatus("Uploading...");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setPdfName(data.pdf_name);
    setUploadStatus(`Uploaded: ${data.pdf_name} (${data.chunks} chunks)`);
  };

  const askQuestion = async () => {
    if (!question.trim() || !pdfName) return;
    setLoading(true);
    setAnswer(null);

    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pdf_name: pdfName, question: question }),
    });
    const data = await res.json();
    setAnswer(data);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: "700px", margin: "40px auto", padding: "20px", fontFamily: "Arial" }}>
      <h1>PDF Chatbot</h1>
      <p>Upload a PDF and ask questions about it.</p>

      {/* Upload section */}
      <div style={{ background: "#f4f4f2", padding: "20px", borderRadius: "8px", marginBottom: "20px" }}>
        <input type="file" accept=".pdf" onChange={handleFileChange} />
        <button
          onClick={uploadPdf}
          style={{ marginLeft: "10px", padding: "8px 16px", backgroundColor: "#2E5FB7", color: "white", border: "none", borderRadius: "6px", cursor: "pointer" }}
        >
          Upload
        </button>
        {uploadStatus && <p style={{ marginTop: "10px", color: "#0F6E56" }}>{uploadStatus}</p>}
      </div>

      {/* Question section */}
      {pdfName && (
        <div>
          <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && askQuestion()}
              placeholder="Ask a question about the PDF..."
              style={{ flex: 1, padding: "10px", fontSize: "16px", borderRadius: "6px", border: "1px solid #ccc" }}
            />
            <button
              onClick={askQuestion}
              style={{ padding: "10px 20px", fontSize: "16px", backgroundColor: "#2E5FB7", color: "white", border: "none", borderRadius: "6px", cursor: "pointer" }}
            >
              Ask
            </button>
          </div>

          {loading && <p>Thinking...</p>}

          {answer && (
            <div style={{ background: "#E1F5EE", padding: "15px", borderRadius: "6px" }}>
              <strong>Answer:</strong>
              <p style={{ margin: "8px 0 0" }}>{answer.answer}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;