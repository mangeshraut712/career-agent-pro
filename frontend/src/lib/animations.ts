import type { Transition } from "framer-motion";

const easeOutExpo: Transition["ease"] = [0.22, 1, 0.36, 1];

export const FADE_IN = {
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, ease: easeOutExpo },
};

export const FADE_UP_delayed = (delay: number) => ({
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, ease: easeOutExpo, delay },
});
