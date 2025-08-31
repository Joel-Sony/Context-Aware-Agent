import { useState } from 'react'

function App() {
  const [message, setMessage] = useState("")
  const [result, setResult] = useState("")

  const handleSubmit = async () =>{
    try{
      const response = await fetch("http://127.0.0.1:5000/submit_message",{
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          user_id:'123',
          message:message
        })
      }) 
      const data = await response.json()
      setResult(data.reply)
    }
    catch(error){
      console.error("Error:",error)
    }
  }
  
  return (
    <div style={{ padding: "20px" }}>
      <label style = {{ fontSize: "3rem"}}>
        Chat with Bot:{" "}
        <input
          type="text"
          style = {{fontSize: "3rem"}}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />
      </label>
      <button 
        onClick={handleSubmit}
        style={{fontSize: "3rem"}}
        >
        Send</button>

      <div style={{fontSize: "3rem"}} >Result: {result}</div>
    </div>
  )
}

export default App
