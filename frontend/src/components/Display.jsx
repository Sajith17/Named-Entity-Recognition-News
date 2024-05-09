import Tag from "./Tag";

function Display({ entities, tagColor, selectedLabels }) {
  return (
    <div className="pl-10 pr-10 mt-8 leading-8 max-w-[700px]">
      {entities.map((item, index) => {
        if (item[1] && selectedLabels[item[1]])
          return (
            <Tag
              key={index}
              string={item[0]}
              label={item[1]}
              tagColor={tagColor}
            />
          );
        else return item[0] + " ";
      })}
    </div>
  );
}

export default Display;
