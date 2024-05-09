import Input from "./Input";
import Display from "./Display";
import { useState, useEffect } from "react";

// const labels = {
//   PERSON: "Person",
//   ORGANIZATION: "Organization",
//   LOCATION: "Location",
//   MISCELLANEOUS: "Miscellaneous",
// };

const tagColor = {
  PERSON: "red",
  ORGANIZATION: "orange",
  LOCATION: "green",
  MISCELLANEOUS: "yellow",
};

const demoSentence =
  "Ukraine launched drone attacks on Russia's Kushchevsk military airfield in the southern Krasnodar region, as well as two oil refineries, a source with knowledge of the operation told CNN.";

// const demoSentence = `Launching an attack against Prime Minister Narendra Modi and the BJP-led central government, Reddy said, "PM Modi and Amit Shah think that only Gujarat is India. That is not the case. There are other states as well. They also have rights," Reddy said.`

const demoEntities = [
  ["Ukraine", "LOCATION"],
  ["launched", null],
  ["drone", null],
  ["attacks", null],
  ["on", null],
  ["Russia's", "LOCATION"],
  ["Kushchevsk", "LOCATION"],
  ["military", null],
  ["airfield", null],
  ["in", null],
  ["the", null],
  ["southern", null],
  ["Krasnodar", "LOCATION"],
  ["region,", null],
  ["as", null],
  ["well", null],
  ["as", null],
  ["two", null],
  ["oil", null],
  ["refineries,", null],
  ["a", null],
  ["source", null],
  ["with", null],
  ["knowledge", null],
  ["of", null],
  ["the", null],
  ["operation", null],
  ["told", null],
  ["CNN.", "ORGANIZATION"],
];

function Container() {
  const [sentence, setSentence] = useState(demoSentence);
  const [entities, setEntities] = useState(demoEntities);
  // const availableLabels = entities
  //   .filter((item) => item[1] !== null)
  //   .reduce((acc, [_, value]) => {
  //     acc[value] = true;
  //     return acc;
  //   }, {});
  const [selectedLabels, setSelectedLabels] = useState({
    PERSON: false,
    ORGANIZATION: false,
    LOCATION: false,
    MISCELLANEOUS: false,
  });
  useEffect(() => {
    const availableLabels = entities
      .filter((item) => item[1] !== null)
      .reduce((acc, [_, value]) => {
        acc[value] = true;
        return acc;
      }, {});
    setSelectedLabels((prevState) => ({ ...prevState, ...availableLabels }));
  }, [entities]);
  return (
    <div className="align-center pl-10 pr-10">
      <Input
        sentence={sentence}
        tagColor={tagColor}
        selectedlabels={selectedLabels}
        setSelectedLabels={setSelectedLabels}
        setSentence={setSentence}
        setEntities={setEntities}
      />
      <Display
        entities={entities}
        tagColor={tagColor}
        selectedLabels={selectedLabels}
      />
    </div>
  );
}

export default Container;
