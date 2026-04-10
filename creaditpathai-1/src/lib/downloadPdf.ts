import html2canvas from "html2canvas";
import jsPDF from "jspdf";

export async function downloadPdf(element: HTMLElement, filename = "credit-report.pdf") {
  // Clone the element to avoid modifying the original
  const clone = element.cloneNode(true) as HTMLElement;
  clone.style.position = "absolute";
  clone.style.left = "-9999px";
  clone.style.top = "0";
  clone.style.width = `${element.offsetWidth}px`;
  clone.style.backgroundColor = "#ffffff";
  clone.style.color = "#000000";
  clone.style.padding = "24px";

  // Force all text to be dark for PDF readability
  clone.querySelectorAll("*").forEach((el) => {
    const htmlEl = el as HTMLElement;
    const computed = window.getComputedStyle(el);
    // Resolve CSS variables to actual colors
    htmlEl.style.backgroundColor = computed.backgroundColor;
    htmlEl.style.color = computed.color;
    htmlEl.style.borderColor = computed.borderColor;
  });

  document.body.appendChild(clone);

  try {
    const canvas = await html2canvas(clone, {
      scale: 2,
      useCORS: true,
      backgroundColor: "#ffffff",
      logging: false,
      windowWidth: element.offsetWidth,
      windowHeight: clone.scrollHeight,
    });

    const imgData = canvas.toDataURL("image/png");
    const imgWidth = 210; // A4 width in mm
    const imgHeight = (canvas.height * imgWidth) / canvas.width;

    const pdf = new jsPDF("p", "mm", "a4");
    let position = 0;
    let remainingHeight = imgHeight;
    const pageHeight = 297; // A4 height in mm

    while (remainingHeight > 0) {
      pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
      remainingHeight -= pageHeight;
      if (remainingHeight > 0) {
        position -= pageHeight;
        pdf.addPage();
      }
    }

    pdf.save(filename);
  } finally {
    document.body.removeChild(clone);
  }
}
