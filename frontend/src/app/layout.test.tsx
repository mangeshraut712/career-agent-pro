import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/layout/Navbar", () => ({
  Navbar: () => <nav>Navigation</nav>,
}));

vi.mock("@/components/layout/Footer", () => ({
  Footer: () => <footer>Footer</footer>,
}));

vi.mock("@/components/ui/Toast", () => ({
  ToastProvider: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

import RootLayout from "./layout";

describe("RootLayout", () => {
  it("uses the bundled system font stack without external font preconnects", () => {
    const markup = renderToStaticMarkup(
      <RootLayout>
        <p>Career workspace</p>
      </RootLayout>,
    );

    expect(markup).toContain("font-sans");
    expect(markup).not.toContain("fonts.googleapis.com");
    expect(markup).not.toContain("fonts.gstatic.com");
  });
});
