export const removeChild = (inputClass) => {
  return new Promise((resolve) => {
    const parent = document.querySelector(inputClass);

    let child = parent.lastElementChild;
    while (child) {
      console.log("child: ", child);
      parent.removeChild(child);
      child = parent.lastElementChild;
    }
    resolve("finished");
  });
};

export const checkArrayMatch = (a, b) => {
  const z = a.map((item) => {
    return JSON.stringify(item);
  });
  return z.includes(JSON.stringify(b));
};
