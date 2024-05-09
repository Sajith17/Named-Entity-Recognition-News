import { Search } from "lucide-react";

function SearchButton({ sentence, setSentence, setEntities }) {
  const fetchData = async (sentence) => {
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sentence: sentence }), // Your JSON data here
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      const data = await response.json();
      setEntities(data.entities);
    } catch (error) {
      console.error("Error:", error);
    }
  };
  return (
    <button
      type="button"
      onClick={() => fetchData(sentence)}
      className="flex justify-center items-center bg-white w-16"
    >
      <Search />
    </button>
  );
}

export default SearchButton;
