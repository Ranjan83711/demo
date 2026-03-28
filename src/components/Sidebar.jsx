const Sidebar = () => {
    return (
        <div className="w-64 h-screen bg-gray-900 text-white p-5">
            <h1 className="text-xl font-bold mb-6">AI Assistant</h1>

            <ul className="space-y-4">
                <li className="hover:text-blue-400 cursor-pointer">💬 Chat</li>
                <li className="hover:text-blue-400 cursor-pointer">📄 Report</li>
                <li className="hover:text-blue-400 cursor-pointer">🩻 X-ray</li>
                <li className="hover:text-blue-400 cursor-pointer">📊 Risk</li>
            </ul>
        </div>
    );
};

export default Sidebar;