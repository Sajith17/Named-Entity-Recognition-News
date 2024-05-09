const colorVariants = {
  red: "bg-red-500",
  orange: "bg-orange-400",
  green: "bg-green-400",
  yellow: "bg-yellow-400",
};

function Tag({ string, label, tagColor }) {
  label = label.toUpperCase();
  return (
    <div
      className={`inline items-center rounded-lg px-2 pb-1 mr-[5px] ${
        colorVariants[tagColor[label]]
      }`}
    >
      <span className="mr-2">{string}</span>
      <span className="bg-white font-bold text-[9px] rounded px-[2px] ">
        {label.slice(0, 3) + " "}
      </span>
    </div>
  );
}

export default Tag;
