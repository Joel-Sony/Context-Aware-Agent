import { useState, useEffect } from "react";
import { Menu, MessageSquare, Plus, Home } from "lucide-react";

const user_id = 123 //TEMPORARY SINGLE USER 

const Start = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  useEffect(()=>{
    const loadMessages = async () =>{
      setIsLoading(true);
      try {
        const response = await fetch('http://127.0.0.1:5000/get_messages',{
          method : "POST",
          headers : {"Content-Type":"application/json"},
          body:JSON.stringify({
            user_id:user_id
          })
        })
    
        const data = await response.json()
        setMessages([
          { sender: "lenni", text: "Hi, I'm Lenni. What's on your mind today?" },
          ...data.messages
        ])
      } catch (error) {
        console.error("Error loading messages:", error);
      } finally {
        setIsLoading(false);
      }
    }

    loadMessages()
  },[])

  
  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { sender: "user", text: input };
    setMessages([...messages, userMessage]);
    setInput("");
    setIsLoading(true);
    
    try {
      const response = await fetch('http://127.0.0.1:5000/submit_message',{
        method : "POST",
        headers : {"Content-Type":"application/json"},
        body:JSON.stringify({
          user_id:user_id,
          message:input
        })
      })
      const data = await response.json()
      const botReply = {
        sender: "lenni",
        text: data.reply
      }
      setMessages(prev => [...prev, botReply]);
    } catch (error) {
      console.error("Error sending message:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen flex bg-gray-50">
      {/* Sidebar */}
      <aside className={`${isSidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-white border-r border-gray-200 flex flex-col overflow-hidden`}>
        {/* <div className="p-4 border-b border-gray-200">
          <button className="w-full flex items-center gap-3 px-4 py-3 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition">
            <Plus size={20} />
            <span>New Chat</span>
          </button>
        </div> */}
        
        <div className="flex-1 overflow-y-auto p-3">
          <div className="space-y-2">
            <div className="px-4 py-3 bg-indigo-50 rounded-lg border-l-4 border-indigo-500 cursor-pointer">
              <div className="flex items-center gap-3">
                <MessageSquare size={18} className="text-indigo-500" />
                <span className="text-sm font-medium text-gray-800">Current Chat</span>
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-gray-200">
          <button 
            onClick={() => window.location.href = '/'}
            className="w-full flex items-center gap-3 px-4 py-3 text-gray-700 hover:bg-gray-100 rounded-lg transition"
          >
            <Home size={20} />
            <span>Home</span>
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-4">
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 hover:bg-gray-100 rounded-lg transition"
          >
            <Menu size={24} className="text-gray-700" />
          </button>
          <div className="flex-1 text-center">
            <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-indigo-500 to-purple-600 bg-clip-text text-transparent">
              Talk to Lenni
            </h1>
            <p className="text-sm text-gray-600 mt-1">
              Your private, judgment-free space
            </p>
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 bg-gray-50">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`w-full px-6 py-4 rounded-2xl text-lg md:text-xl shadow-sm ${
                  msg.sender === "user"
                    ? "bg-indigo-500 text-white ml-auto"
                    : "bg-white text-gray-800 border border-gray-200"
                }`}
              >
                {msg.text}
              </div>
            ))}
            
            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex items-center gap-3 px-6 py-4 bg-white rounded-2xl border border-gray-200 shadow-sm">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-indigo-500"></div>
                <span className="text-gray-600">Lenni is typing...</span>
              </div>
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-6">
          <form
            onSubmit={handleSend}
            className="max-w-4xl mx-auto flex items-center gap-4"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
              placeholder="Type your message..."
              className="flex-1 px-6 py-4 text-lg md:text-xl rounded-full border-2 border-gray-200 focus:outline-none focus:border-indigo-400 transition disabled:bg-gray-50 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-8 py-4 text-lg bg-indigo-500 text-white rounded-full hover:bg-indigo-600 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg"
            >
              {isLoading ? (
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
              ) : (
                "Send"
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Start;