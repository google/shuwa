const initHelpmodal = () => {
  const containerEl = document.querySelector(".help-modal-container");
  const modalEl = document.querySelector(".help-modal");
  const helpBtn = document.querySelector("#help-btn");
  const closeBtn = document.querySelector("#help-modal-close");

  helpBtn.addEventListener("click", () => {
    modalEl.style.transform = `translate(0, 0)`;
    containerEl.style.background = `rgba(0, 0, 0, 0.25)`;
    containerEl.style.visibility = "visible";
  });

  closeBtn.addEventListener("click", () => {
    modalEl.style.transform = `translate(0, 200%)`;
    containerEl.style.background = `rgba(0, 0, 0, 0)`;
    containerEl.style.visibility = "hidden";
  });
};

export default initHelpmodal;
