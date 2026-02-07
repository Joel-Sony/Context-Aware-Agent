import { useState, useEffect, useRef } from "react";
import { Menu, MessageSquare, Plus, Home, Mic, Play, Pause, X } from "lucide-react";
import ReactMarkdown from 'react-markdown';

const user_id = 123; //TEMPORARY SINGLE USER 
const API_BASE_URL = 'http://127.0.0.1:5000';

const Start = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  // Voice recording states
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState(null);
  const [playingId, setPlayingId] = useState(null);
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioRefs = useRef({});

  useEffect(() => {
    const loadMessages = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/get_messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_id: user_id
          })
        });
    
        const data = await response.json();
        
        // Process messages and add audio URLs for voice messages
        const processedMessages = data.messages.map(msg => {
          if (msg.type === "voice" && msg.audio_filename) {
            return {
              ...msg,
              audioUrl: `${API_BASE_URL}/audio/${msg.audio_filename}`,
              duration: Math.round(msg.audio_duration || 0)
            };
          }
          return msg;
        });
        
        setMessages([
          { sender: "lenni", text: "Hi, I'm Lenni. What's on your mind today?", type: "text" },
          ...processedMessages
        ]);
      } catch (error) {
        console.error("Error loading messages:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadMessages();
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Could not access microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const cancelRecording = () => {
    stopRecording();
    setAudioBlob(null);
    setRecordingTime(0);
    audioChunksRef.current = [];
  };

  const sendVoiceMessage = async () => {
    if (!audioBlob) return;

    const url = URL.createObjectURL(audioBlob);
    const userMessage = {
      id: Date.now(),
      sender: "user",
      type: "voice",
      audioUrl: url,
      duration: recordingTime
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send audio to backend
      const formData = new FormData();
      formData.append('audio', audioBlob, 'voice_message.webm');
      formData.append('user_id', user_id);

      const response = await fetch(`${API_BASE_URL}/submit_voice_message`, {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      
      // Update the user message with the server audio URL
      setMessages(prev => {
        const updated = [...prev];
        const lastUserMsg = updated[updated.length - 1];
        if (lastUserMsg.sender === "user" && lastUserMsg.type === "voice") {
          lastUserMsg.audioUrl = `${API_BASE_URL}/audio/${data.audio_filename}`;
          lastUserMsg.duration = Math.round(data.duration);
        }
        return updated;
      });
      
      const botReply = {
        sender: "lenni",
        text: data.reply,
        type: "text"
      };
      setMessages(prev => [...prev, botReply]);
    } catch (error) {
      console.error("Error sending voice message:", error);
    } finally {
      setIsLoading(false);
      setAudioBlob(null);
      setRecordingTime(0);
    }
  };

  const togglePlayPause = (id, url) => {
    const audio = audioRefs.current[id];
    
    if (playingId === id) {
      audio.pause();
      setPlayingId(null);
    } else {
      Object.values(audioRefs.current).forEach(a => a.pause());
      
      if (!audio) {
        const newAudio = new Audio(url);
        audioRefs.current[id] = newAudio;
        newAudio.onended = () => setPlayingId(null);
        newAudio.play();
      } else {
        audio.play();
      }
      setPlayingId(id);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { sender: "user", text: input, type: "text" };
    setMessages([...messages, userMessage]);
    setInput("");
    setIsLoading(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/submit_message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: user_id,
          message: input
        })
      });
      const data = await response.json();
      const botReply = {
        sender: "lenni",
        text: data.reply,
        type: "text"
      };
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
      <div className="flex-1 flex flex-col relative">
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
                className={`w-full px-6 py-4 rounded-2xl shadow-sm ${
                  msg.sender === "user"
                    ? "bg-indigo-500 text-white ml-auto"
                    : "bg-white text-gray-800 border border-gray-200"
                }`}
              >
                {msg.type === "voice" ? (
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => togglePlayPause(msg.id || idx, msg.audioUrl)}
                      className={`rounded-full p-2 transition ${
                        msg.sender === "user"
                          ? "bg-white text-indigo-500 hover:bg-indigo-50"
                          : "bg-indigo-500 text-white hover:bg-indigo-600"
                      }`}
                    >
                      {playingId === (msg.id || idx) ? (
                        <Pause size={20} />
                      ) : (
                        <Play size={20} />
                      )}
                    </button>
                    <div className="flex-1">
                      <div className={`h-1 rounded-full ${
                        msg.sender === "user" ? "bg-indigo-300" : "bg-indigo-200"
                      }`}></div>
                    </div>
                    <span className={`text-sm ${
                      msg.sender === "user" ? "text-indigo-100" : "text-gray-600"
                    }`}>
                      {formatTime(msg.duration || 0)}
                    </span>
                  </div>
                ) : (
                  <div className="prose prose-slate max-w-none text-lg md:text-xl">
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                )}
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

        {/* Recording Overlay - positioned over input area */}
        {isRecording && (
          <div className="absolute bottom-0 left-0 right-0 bg-white border-t-2 border-red-500 p-6 shadow-2xl z-50">
            <div className="max-w-4xl mx-auto">
              <div className="flex items-center justify-between bg-red-50 rounded-lg p-4">
                <div className="flex items-center gap-4">
                  <div className="relative w-12 h-12">
                    <div className="absolute inset-0 bg-red-500 rounded-full animate-pulse"></div>
                    <div className="absolute inset-1 bg-white rounded-full flex items-center justify-center">
                      <Mic className="text-red-500" size={20} />
                    </div>
                  </div>
                  <div>
                    <p className="text-xl font-bold text-gray-800">
                      {formatTime(recordingTime)}
                    </p>
                    <p className="text-sm text-gray-600">Recording...</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={cancelRecording}
                    className="bg-gray-200 text-gray-700 px-6 py-2 rounded-lg hover:bg-gray-300 transition font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={stopRecording}
                    className="bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-600 transition font-medium"
                  >
                    Stop
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Preview Recording */}
        {audioBlob && !isRecording && (
          <div className="bg-yellow-50 border-t border-yellow-200 px-6 py-4">
            <div className="max-w-4xl mx-auto flex items-center gap-4">
              <div className="flex-1 flex items-center gap-3 bg-white rounded-lg px-4 py-3 shadow">
                <Mic className="text-indigo-500" size={20} />
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-700">Voice message ready</p>
                  <p className="text-xs text-gray-500">{formatTime(recordingTime)}</p>
                </div>
              </div>
              <button
                onClick={cancelRecording}
                className="bg-gray-200 text-gray-700 p-3 rounded-full hover:bg-gray-300 transition"
              >
                <X size={20} />
              </button>
              <button
                onClick={sendVoiceMessage}
                disabled={isLoading}
                className="bg-indigo-500 text-white px-8 py-3 rounded-full hover:bg-indigo-600 transition shadow-lg disabled:opacity-50"
              >
                Send
              </button>
            </div>
          </div>
        )}

        {/* Input Area */}
        {!audioBlob && (
          <div className="bg-white border-t border-gray-200 p-6">
            <div className="max-w-4xl mx-auto flex items-center gap-4">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend(e);
                  }
                }}
                disabled={isLoading}
                placeholder="Type your message..."
                className="flex-1 px-6 py-4 text-lg md:text-xl rounded-full border-2 border-gray-200 focus:outline-none focus:border-indigo-400 transition disabled:bg-gray-50 disabled:cursor-not-allowed !w-auto"
                style={{ width: 'auto' }}
              />
              <button
                type="button"
                onClick={startRecording}
                disabled={isLoading || isRecording}
                className="p-4 bg-indigo-500 text-white rounded-full hover:bg-indigo-600 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg flex-shrink-0"
              >
                <Mic size={24} />
              </button>
              <button
                type="button"
                onClick={handleSend}
                disabled={isLoading || !input.trim()}
                className="px-8 py-4 text-lg bg-indigo-500 text-white rounded-full hover:bg-indigo-600 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg flex-shrink-0"
              >
                {isLoading ? (
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                ) : (
                  "Send"
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Start;