import type { AnchorHTMLAttributes, ReactNode } from "react";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Navbar } from "@/components/layout/Navbar";

vi.mock("next/link", () => ({
    default: ({
        children,
        href,
        ...props
    }: {
        children: ReactNode;
        href: string;
    } & AnchorHTMLAttributes<HTMLAnchorElement>) => (
        <a href={href} {...props}>
            {children}
        </a>
    ),
}));

vi.mock("next/navigation", () => ({
    usePathname: () => "/",
}));

describe("Navbar", () => {
    beforeEach(() => {
        vi.restoreAllMocks();
    });

    it("names the profile link and listens for scroll passively", () => {
        const addEventListener = vi.spyOn(window, "addEventListener");
        render(<Navbar />);

        expect(screen.getByRole("link", { name: "Profile" })).toHaveAttribute("href", "/profile");
        expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();

        const scrollListener = addEventListener.mock.calls.find((call) => call[0] === "scroll");
        expect(scrollListener?.[2]).toMatchObject({ passive: true });
    });

    it("exposes the mobile menu state", () => {
        render(<Navbar />);
        const menu = screen.getByRole("button", { name: "Open menu" });
        expect(menu).toHaveAttribute("aria-expanded", "false");
        expect(menu).toHaveAttribute("aria-controls", "mobile-nav");
    });
});
