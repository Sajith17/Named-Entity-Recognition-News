function Label({ name, tagColor, selectedLabels, setSelectedLabels }) {
  const toggeleCheckBox = () => {
    let newSelectedLabels = {
      ...selectedLabels,
      [name]: !selectedLabels[name],
    };
    setSelectedLabels(newSelectedLabels);
  };

  const colorVariants = {
    red: "bg-red-500 hover:bg-red-600 ",
    orange: "bg-orange-400 hover:bg-orange-500",
    green: "bg-green-400 hover:bg-green-500",
    yellow: "bg-yellow-400 hover:bg-yellow-500",
  };

  return (
    <div
      onClick={toggeleCheckBox}
      className={`flex p-1.5 h-7 mr-3 cursor-pointer rounded-lg ${
        colorVariants[tagColor[name]]
      }`}
    >
      <input
        type="checkbox"
        checked={selectedLabels[name]}
        className="mr-1"
        readOnly
      />
      <div className="flex items-center font-bold text-xs select-none">
        {name}
      </div>
    </div>
  );
}

export default Label;
