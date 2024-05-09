import Label from "./Label";
import SearchButton from "./SearchButton";
function Input({
  sentence,
  tagColor,
  selectedlabels,
  setSelectedLabels,
  setSentence,
  setEntities,
}) {
  const handleSentenceChange = (event) => {
    setSentence(event.target.value);
  };
  return (
    <div className="mt-14 bg-blue-600 pt-10 pb-8 px-10">
      <h1 className="mb-7 fs-9 text-white text-2xl font-bold">
        Named Entity Recognition
      </h1>
      <div className="flex flex-wrap">
        <div className="flex grow-[2] mr-10 mb-2">
          <textarea
            name="textinput"
            id="textinput"
            className="p-3 grow text-s min-h-40 resize-none"
            value={sentence}
            onChange={handleSentenceChange}
          ></textarea>
          <SearchButton
            sentence={sentence}
            setSentence={setSentence}
            setEntities={setEntities}
          />
        </div>
        <div className="flex flex-col grow">
          <p className="text-white mb-1 text-xs font-bold">Entity Labels</p>
          <div className="flex grow flex-wrap">
            <Label
              name="PERSON"
              tagColor={tagColor}
              selectedLabels={selectedlabels}
              setSelectedLabels={setSelectedLabels}
            />
            <Label
              name="ORGANIZATION"
              tagColor={tagColor}
              selectedLabels={selectedlabels}
              setSelectedLabels={setSelectedLabels}
            />
            <Label
              name="LOCATION"
              tagColor={tagColor}
              selectedLabels={selectedlabels}
              setSelectedLabels={setSelectedLabels}
            />
            <Label
              name="MISCELLANEOUS"
              tagColor={tagColor}
              selectedLabels={selectedlabels}
              setSelectedLabels={setSelectedLabels}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default Input;
