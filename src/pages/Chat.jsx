import { useState } from "react";
import { useNavigate } from "react-router-dom";

const Chat = () => {
    const navigate = useNavigate();
    const [message, setMessage] = useState("");
    const [chat, setChat] = useState([]);

    const sendMessage = async () => {
        if (!message.trim()) return;

        // User message add
        const newChat = [...chat, { role: "user", text: message }];
        setChat(newChat);

        // Backend call (your /ask API)
        try {
            const res = await fetch("http://127.0.0.1:8000/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query: message })
            });

            const data = await res.json();

            setChat([
                ...newChat,
                { role: "bot", text: data.answer || "No response" }
            ]);

        } catch (err) {
            setChat([
                ...newChat,
                { role: "bot", text: "⚠️ Error connecting to server" }
            ]);
        }

        setMessage("");
    };

    return (
        <div className="h-screen flex flex-col">

            {/* Header */}
            <div className="p-4 bg-teal-500 text-white flex items-center">
                <button onClick={() => navigate("/")} className="mr-4">
                    ⬅
                </button>
                <h1 className="text-xl font-semibold">AI Chat</h1>
            </div>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-100">
                {chat.map((msg, index) => (
                    <div
                        key={index}
                        className={`max-w-xl p-3 rounded-lg ${msg.role === "user"
                                ? "bg-teal-500 text-white ml-auto"
                                : "bg-white text-black"
                            }`}
                    >
                        {msg.text}
                    </div>
                ))}
            </div>

            {/* Input */}
            <div className="p-4 bg-white flex gap-2">
                <input
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-1 border rounded-lg px-4 py-2"
                />
                <button
                    onClick={sendMessage}
                    className="bg-teal-500 text-white px-4 py-2 rounded-lg"
                >
                    Send
                </button>
            </div>

        </div>
    );
};

export default Chat;