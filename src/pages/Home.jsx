import { useNavigate } from "react-router-dom";

const Home = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen bg-gradient-to-br from-teal-100 to-white p-6">

            {/* Header */}
            <h1 className="text-3xl font-bold text-center mb-6">
                🤖 AI Medical Assistant
            </h1>

            {/* Welcome */}
            <div className="mb-6">
                <h2 className="text-xl font-semibold">Welcome, User 👋</h2>
                <p className="text-gray-600">How can I help you today?</p>
            </div>

            {/* Cards */}
            <div className="flex justify-center">
                <div className="grid grid-cols-2 md:grid-cols-5 gap-6">

                    {/* Voice */}
                    <div className="bg-white w-32 h-32 flex flex-col items-center justify-center rounded-2xl shadow-md cursor-pointer">
                        🎤
                        <p>Voice Chat</p>
                    </div>

                    {/* Text Chat */}
                    <div
                        onClick={() => navigate("/chat")}
                        className="bg-white w-32 h-32 flex flex-col items-center justify-center rounded-2xl shadow-md cursor-pointer"
                    >
                        💬
                        <p>Text Chat</p>
                    </div>

                    {/* OCR */}
                    <div className="bg-white w-32 h-32 flex flex-col items-center justify-center rounded-2xl shadow-md">
                        📄
                        <p>OCR Report</p>
                    </div>

                    {/* Scan */}
                    <div className="bg-white w-32 h-32 flex flex-col items-center justify-center rounded-2xl shadow-md">
                        🩻
                        <p>Scan Report</p>
                    </div>

                    {/* Risk */}
                    <div className="bg-white w-32 h-32 flex flex-col items-center justify-center rounded-2xl shadow-md">
                        ❤️
                        <p>Risk Prediction</p>
                    </div>

                </div>
            </div>

            {/* Bottom box */}
            <div className="mt-10 bg-white rounded-xl shadow p-6 text-center max-w-xl mx-auto">
                <p className="text-gray-500">Select a service to begin</p>
            </div>

        </div>
    );
};

export default Home;